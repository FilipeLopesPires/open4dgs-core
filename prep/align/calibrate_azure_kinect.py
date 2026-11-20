#!/usr/bin/env python3
"""Solve multi-camera extrinsics for an Azure Kinect capture using colour + depth.

Run after ``prep/extract/extract_azure_kinect.py extract``.  Azure Kinect recordings
carry factory intrinsics but no camera-to-camera poses, so this script recovers them
from the capture itself:

1. SIFT features on the undistorted colour frames at several synchronised instants.
2. Each keypoint is lifted to metric 3D with *its own* camera's depth image
   (raw depth reprojected through the factory depth->colour calibration).
3. Every camera pair: mutual ratio-test matches -> PnP-RANSAC (3D in one camera,
   2D in the other).  Pair strength = inlier count.
4. Initial poses from a maximum spanning tree rooted at the reference camera.
5. Joint bundle adjustment over poses and 3D points: pixel reprojection error plus a
   weak depth prior (keeps metric scale), Huber-robustified, with outlier rejection.
6. World frame: floor plane detected from fused depth becomes y = 0, +Y up, origin
   under the point the cameras look at. Open the result in the trainer's viewer with
   ``--up +y`` (its default orbit up axis is viser's +Z).

Outputs in ``--data-dir``::

    sparse/0/{cameras,images,points3D}.txt   PINHOLE COLMAP text model
    sparse/scene_meta.json
    reference_export.json                    for prep-roma / sync tools
    k4a_report/calibration.json              per-pair / per-camera / depth QA
    k4a_report/calibration_fused.ply         fused coloured depth cloud (one frame)
    k4a_report/calibration_reprojection.jpg  other cameras' depth splatted into each view

Example::

    python prep/align/calibrate_azure_kinect.py --data-dir datasets/professor/o4d
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extract"))
from k4a_mkv import Intrinsics  # noqa: E402


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
@dataclass
class Cam:
    folder: str
    source: str
    K: np.ndarray
    width: int
    height: int
    d_intr: Intrinsics | None
    R_dc: np.ndarray
    t_dc: np.ndarray


def load_cameras(data_dir: Path) -> tuple[list[Cam], dict]:
    meta = json.loads((data_dir / "k4a_cameras.json").read_text(encoding="utf-8"))
    cams = []
    for folder in sorted(meta["cameras"]):
        c = meta["cameras"][folder]
        di = c.get("depth_intrinsics")
        d_intr = None
        if di:
            d_intr = Intrinsics(di["width"], di["height"], di["fx"], di["fy"], di["cx"], di["cy"],
                                np.asarray(di["dist_opencv_rational"], dtype=np.float64))
        cams.append(Cam(folder, c["source"], np.asarray(c["K"], dtype=np.float64), int(c["width"]), int(c["height"]),
                        d_intr, np.asarray(c["depth_to_color_R"]), np.asarray(c["depth_to_color_t"])))
    return cams, meta


def depth_in_color(cam: Cam, depth_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sparse z-buffered depth (m) in the output colour camera, and its 3D points (colour frame)."""
    v, u = np.nonzero(depth_mm > 0)
    z = depth_mm[v, u].astype(np.float64) / 1000.0
    xy = cv2.undistortPoints(np.stack([u, v], 1).astype(np.float64)[:, None], cam.d_intr.K, cam.d_intr.dist)[:, 0]
    Xd = np.c_[xy * z[:, None], z]
    Xc = Xd @ cam.R_dc.T + cam.t_dc
    zc = Xc[:, 2]
    keep = zc > 0.1
    Xc = Xc[keep]
    zc = zc[keep]
    pu = np.round(cam.K[0, 0] * Xc[:, 0] / zc + cam.K[0, 2]).astype(np.int64)
    pv = np.round(cam.K[1, 1] * Xc[:, 1] / zc + cam.K[1, 2]).astype(np.int64)
    ok = (pu >= 0) & (pu < cam.width) & (pv >= 0) & (pv < cam.height)
    zmap = np.full((cam.height, cam.width), np.inf, dtype=np.float32)
    np.minimum.at(zmap, (pv[ok], pu[ok]), zc[ok].astype(np.float32))
    return zmap, Xc


def sample_depth(zmap: np.ndarray, pts: np.ndarray, win: int = 7, edge_ratio: float = 1.04) -> np.ndarray:
    """Depth at keypoints: min over a window, rejected at depth discontinuities."""
    z = zmap.copy()
    finite = np.isfinite(z)
    zmin = cv2.erode(np.where(finite, z, 1e6).astype(np.float32), np.ones((win, win), np.uint8))
    zmax = cv2.dilate(np.where(finite, z, 0).astype(np.float32), np.ones((win, win), np.uint8))
    u = np.clip(np.round(pts[:, 0]).astype(int), 0, z.shape[1] - 1)
    v = np.clip(np.round(pts[:, 1]).astype(int), 0, z.shape[0] - 1)
    lo, hi = zmin[v, u], zmax[v, u]
    out = np.where((lo < 1e5) & (hi > 0) & (hi / np.maximum(lo, 1e-6) < edge_ratio), lo, np.nan)
    return out


def backproject(K: np.ndarray, pts: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = (pts[:, 0] - K[0, 2]) / K[0, 0]
    y = (pts[:, 1] - K[1, 2]) / K[1, 1]
    return np.c_[x * z, y * z, z]


# --------------------------------------------------------------------------------------
# Features and pair poses
# --------------------------------------------------------------------------------------
@dataclass
class Feat:
    pts: np.ndarray  # [n,2]
    desc: np.ndarray  # [n,128]
    X: np.ndarray  # [n,3] camera-frame 3D (nan where no depth)
    z: np.ndarray  # [n]


def extract_features(img: np.ndarray, zmap: np.ndarray, K: np.ndarray, n_features: int) -> Feat:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02)
    kps, desc = sift.detectAndCompute(gray, None)
    pts = np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)
    if desc is None:
        desc = np.zeros((0, 128), np.float32)
    # RootSIFT
    desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
    desc = np.sqrt(desc).astype(np.float32)
    z = sample_depth(zmap, pts)
    return Feat(pts, desc, backproject(K, pts, z), z)


def match(a: Feat, b: Feat, ratio: float) -> np.ndarray:
    if len(a.desc) < 2 or len(b.desc) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ab = bf.knnMatch(a.desc, b.desc, k=2)
    ba = bf.knnMatch(b.desc, a.desc, k=2)
    best_ba = {m[0].queryIdx: m[0].trainIdx for m in ba if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    out = [(m[0].queryIdx, m[0].trainIdx) for m in ab
           if len(m) == 2 and m[0].distance < ratio * m[1].distance and best_ba.get(m[0].trainIdx) == m[0].queryIdx]
    return np.array(out, dtype=int).reshape(-1, 2)


def pnp(X: np.ndarray, uv: np.ndarray, K: np.ndarray, thresh: float):
    if len(X) < 12:
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, uv, K, None, iterationsCount=10000, reprojectionError=thresh,
                                             confidence=0.9999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 12:
        return None
    inl = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(X[inl], uv[inl], K, None, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    return R, tvec[:, 0], inl


# --------------------------------------------------------------------------------------
# Bundle adjustment
# --------------------------------------------------------------------------------------
class UnionFind:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def bundle_adjust(Rs, ts, cams, obs_cam, obs_pt, obs_uv, obs_z, X0, ref, depth_sigma, max_nfev, log):
    n_cam, n_pt = len(cams), len(X0)
    free = [c for c in range(n_cam) if c != ref]
    cidx = {c: i for i, c in enumerate(free)}
    x0 = [np.r_[Rotation.from_matrix(Rs[c]).as_rotvec(), ts[c]] for c in free]
    x0 = np.concatenate(x0 + [X0.ravel()])
    Ks = np.stack([c.K for c in cams])
    has_z = np.isfinite(obs_z)
    n_obs = len(obs_cam)
    n_res = 2 * n_obs + int(has_z.sum())
    zrows = np.flatnonzero(has_z)
    zs = np.maximum(depth_sigma * obs_z[has_z] ** 2, depth_sigma * 0.5)  # ToF noise grows ~ z^2

    def unpack(x):
        R = np.array(Rs, dtype=np.float64).copy()
        t = np.array(ts, dtype=np.float64).copy()
        for c in free:
            p = x[6 * cidx[c] : 6 * cidx[c] + 6]
            R[c] = Rotation.from_rotvec(p[:3]).as_matrix()
            t[c] = p[3:]
        X = x[6 * len(free) :].reshape(-1, 3)
        return R, t, X

    def fun(x):
        R, t, X = unpack(x)
        Xc = np.einsum("nij,nj->ni", R[obs_cam], X[obs_pt]) + t[obs_cam]
        z = np.maximum(Xc[:, 2], 1e-3)
        K = Ks[obs_cam]
        u = K[:, 0, 0] * Xc[:, 0] / z + K[:, 0, 2]
        v = K[:, 1, 1] * Xc[:, 1] / z + K[:, 1, 2]
        r = np.empty(n_res)
        r[0 : 2 * n_obs : 2] = u - obs_uv[:, 0]
        r[1 : 2 * n_obs : 2] = v - obs_uv[:, 1]
        r[2 * n_obs :] = (z[has_z] - obs_z[has_z]) / zs
        return r

    A = lil_matrix((n_res, len(x0)), dtype=np.int8)
    base_pt = 6 * len(free)
    for k in range(n_obs):
        c, p = obs_cam[k], obs_pt[k]
        rows = [2 * k, 2 * k + 1]
        for row in rows:
            if c in cidx:
                A[row, 6 * cidx[c] : 6 * cidx[c] + 6] = 1
            A[row, base_pt + 3 * p : base_pt + 3 * p + 3] = 1
    for j, k in enumerate(zrows):
        c, p = obs_cam[k], obs_pt[k]
        row = 2 * n_obs + j
        if c in cidx:
            A[row, 6 * cidx[c] : 6 * cidx[c] + 6] = 1
        A[row, base_pt + 3 * p : base_pt + 3 * p + 3] = 1
    log(f"  BA: {len(free)} free cameras, {n_pt} points, {n_obs} observations ({int(has_z.sum())} with depth)")
    res = least_squares(fun, x0, jac_sparsity=A.tocsr(), loss="huber", f_scale=2.0, x_scale="jac",
                        method="trf", max_nfev=max_nfev, verbose=0)
    R, t, X = unpack(res.x)
    r = fun(res.x)
    reproj = np.hypot(r[0 : 2 * n_obs : 2], r[1 : 2 * n_obs : 2])
    dres = np.full(n_obs, np.nan)
    dres[zrows] = r[2 * n_obs :] * zs
    return R, t, X, reproj, dres


# --------------------------------------------------------------------------------------
# World frame
# --------------------------------------------------------------------------------------
def fit_floor(points: np.ndarray, up_hint: np.ndarray, cam_centers: np.ndarray, iters: int = 3000, thresh: float = 0.02):
    """RANSAC plane whose normal is within 25 deg of up_hint, below every camera, maximising support."""
    rng = np.random.default_rng(0)
    best = (0, None)
    n = len(points)
    for _ in range(iters):
        s = points[rng.choice(n, 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        L = np.linalg.norm(nrm)
        if L < 1e-9:
            continue
        nrm /= L
        if nrm @ up_hint < 0:
            nrm = -nrm
        if nrm @ up_hint < math.cos(math.radians(25)):
            continue
        d = -nrm @ s[0]
        if np.any(cam_centers @ nrm + d < 0.3):  # cameras must be above the floor
            continue
        support = int((np.abs(points @ nrm + d) < thresh).sum())
        if support > best[0]:
            best = (support, (nrm, d))
    if best[1] is None:
        return None
    nrm, d = best[1]
    inl = points[np.abs(points @ nrm + d) < thresh]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    nrm = vt[2] if vt[2] @ up_hint > 0 else -vt[2]
    return nrm, -nrm @ c, len(inl)


def look_at_point(centers: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Least-squares point closest to all optical axes."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centers, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def rotation_to_qvec(R: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()  # x y z w
    q = np.r_[q[3], q[:3]]
    return q if q[0] >= 0 else -q


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--reference-camera", default=None, help="camera folder fixed at the origin (default: first)")
    ap.add_argument("--num-sample-frames", type=int, default=8, help="synchronised instants used for matching")
    ap.add_argument("--sift-features", type=int, default=12000)
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--pnp-threshold", type=float, default=4.0, help="RANSAC reprojection threshold (px)")
    ap.add_argument("--min-pair-inliers", type=int, default=40)
    ap.add_argument("--depth-sigma", type=float, default=0.004,
                    help="depth prior: sigma = depth_sigma * z^2 metres (weak by default)")
    ap.add_argument("--outlier-px", type=float, default=4.0, help="drop observations above this after the first BA")
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--world-frame", choices=("floor", "reference"), default="floor",
                    help="floor: y=0 floor, +Y up, origin under the look-at point; reference: reference camera frame")
    ap.add_argument("--image-ext", default="jpg")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    data = Path(args.data_dir)
    report_dir = data / "k4a_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    sparse = data / "sparse" / "0"
    if sparse.exists() and any(sparse.iterdir()) and not args.overwrite:
        raise SystemExit(f"{sparse} exists; pass --overwrite")
    cams, meta = load_cameras(data)
    n_cam = len(cams)
    folders = [c.folder for c in cams]
    ref = folders.index(args.reference_camera) if args.reference_camera else 0
    if any(c.d_intr is None for c in cams):
        raise SystemExit("every camera needs depth for metric calibration (extract without --no-depth)")
    n_frames = int(meta["num_frames"])
    frames = np.unique(np.linspace(1, n_frames, args.num_sample_frames).round().astype(int))
    print(f"{n_cam} cameras, sampling frames {frames.tolist()}")

    # ---- features ---------------------------------------------------------------------
    t0 = time.time()
    feats: dict[tuple[int, int], Feat] = {}
    for f in frames:
        for ci, cam in enumerate(cams):
            img = cv2.imread(str(data / "images" / cam.folder / f"{f:06d}.{args.image_ext}"))
            dep = cv2.imread(str(data / "depth" / cam.folder / f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
            if img is None or dep is None:
                raise SystemExit(f"missing image/depth for camera {cam.folder} frame {f}")
            zmap, _ = depth_in_color(cam, dep)
            feats[(f, ci)] = extract_features(img, zmap, cam.K, args.sift_features)
    print(f"Features in {time.time() - t0:.1f} s (median {int(np.median([len(v.pts) for v in feats.values()]))} per image)")

    # ---- pairwise -------------------------------------------------------------------------
    pair_matches: dict[tuple[int, int], list] = {}
    pair_info = {}
    for a in range(n_cam):
        for b in range(a + 1, n_cam):
            per = []
            for f in frames:
                m = match(feats[(f, a)], feats[(f, b)], args.ratio)
                per.append((f, m))
            # PnP both directions on pooled matches
            Xa = np.concatenate([feats[(f, a)].X[m[:, 0]] for f, m in per])
            ua = np.concatenate([feats[(f, a)].pts[m[:, 0]] for f, m in per])
            Xb = np.concatenate([feats[(f, b)].X[m[:, 1]] for f, m in per])
            ub = np.concatenate([feats[(f, b)].pts[m[:, 1]] for f, m in per])
            best = None
            okA = np.isfinite(Xa[:, 2])
            r = pnp(Xa[okA], ub[okA], cams[b].K, args.pnp_threshold)  # T_b<-a
            if r is not None:
                best = (len(r[2]), r[0], r[1])
            okB = np.isfinite(Xb[:, 2])
            r2 = pnp(Xb[okB], ua[okB], cams[a].K, args.pnp_threshold)  # T_a<-b
            if r2 is not None and (best is None or len(r2[2]) > best[0]):
                Rab, tab = r2[0], r2[1]
                best = (len(r2[2]), Rab.T, -Rab.T @ tab)
            n_m = sum(len(m) for _, m in per)
            pair_info[f"{folders[a]}-{folders[b]}"] = {"matches": int(n_m), "pnp_inliers": int(best[0]) if best else 0}
            if best and best[0] >= args.min_pair_inliers:
                pair_matches[(a, b)] = [per, best]
            print(f"  pair {folders[a]}-{folders[b]}: {n_m:6d} matches, {best[0] if best else 0:5d} PnP inliers")

    # ---- spanning tree --------------------------------------------------------------------
    Rs = [None] * n_cam
    ts = [None] * n_cam
    Rs[ref], ts[ref] = np.eye(3), np.zeros(3)
    done = {ref}
    while len(done) < n_cam:
        cand = []
        for (a, b), (_, best) in pair_matches.items():
            if (a in done) != (b in done):
                cand.append((best[0], a, b, best))
        if not cand:
            missing = [folders[i] for i in range(n_cam) if i not in done]
            raise SystemExit(f"cameras {missing} have no reliable pair to the rest; inspect k4a_report")
        _, a, b, best = max(cand, key=lambda c: c[0])
        R_ba, t_ba = best[1], best[2]
        if a in done:  # T_b = T_ba T_a
            Rs[b] = R_ba @ Rs[a]
            ts[b] = R_ba @ ts[a] + t_ba
            done.add(b)
        else:  # T_a = T_ba^-1 T_b
            Rab, tab = R_ba.T, -R_ba.T @ t_ba
            Rs[a] = Rab @ Rs[b]
            ts[a] = Rab @ ts[b] + tab
            done.add(a)

    # ---- tracks ---------------------------------------------------------------------------
    uf = UnionFind()
    for (a, b), (per, _) in pair_matches.items():
        for f, m in per:
            for ia, ib in m:
                uf.union((f, a, int(ia)), (f, b, int(ib)))
    groups: dict = {}
    for node in list(uf.p):
        groups.setdefault(uf.find(node), []).append(node)
    obs_cam, obs_pt, obs_uv, obs_z, X0 = [], [], [], [], []
    for nodes in groups.values():
        if len(nodes) < 2 or len({n[1] for n in nodes}) != len(nodes):  # one obs per camera
            continue
        world = []
        for f, c, i in nodes:
            X = feats[(f, c)].X[i]
            if np.isfinite(X[2]):
                world.append(Rs[c].T @ (X - ts[c]))
        if not world:
            continue
        world = np.array(world)
        Xw = np.median(world, axis=0)
        # initial consistency check
        errs = []
        for f, c, i in nodes:
            Xc = Rs[c] @ Xw + ts[c]
            if Xc[2] <= 0.1:
                errs.append(1e9)
                continue
            uv = cams[c].K @ (Xc / Xc[2])
            errs.append(np.hypot(*(uv[:2] - feats[(f, c)].pts[i])))
        if np.median(errs) > 4 * args.pnp_threshold:
            continue
        pid = len(X0)
        X0.append(Xw)
        for (f, c, i), e in zip(nodes, errs):
            if e > 4 * args.pnp_threshold:
                continue
            obs_cam.append(c)
            obs_pt.append(pid)
            obs_uv.append(feats[(f, c)].pts[i])
            obs_z.append(feats[(f, c)].z[i])
    obs_cam = np.array(obs_cam)
    obs_pt = np.array(obs_pt)
    obs_uv = np.array(obs_uv)
    obs_z = np.array(obs_z)
    X0 = np.array(X0)
    print(f"Tracks: {len(X0)} points, {len(obs_cam)} observations")

    # ---- BA (two rounds with outlier rejection) --------------------------------------------
    R, t, X, reproj, dres = bundle_adjust(Rs, ts, cams, obs_cam, obs_pt, obs_uv, obs_z, X0, ref,
                                          args.depth_sigma, args.max_nfev, print)
    print(f"  round 1: median reprojection {np.median(reproj):.2f} px, p90 {np.percentile(reproj, 90):.2f} px")
    keep = reproj < args.outlier_px
    # drop points left with fewer than 2 observations
    cnt = np.bincount(obs_pt[keep], minlength=len(X))
    keep &= cnt[obs_pt] >= 2
    remap = -np.ones(len(X), int)
    used = np.unique(obs_pt[keep])
    remap[used] = np.arange(len(used))
    obs_cam, obs_pt, obs_uv, obs_z = obs_cam[keep], remap[obs_pt[keep]], obs_uv[keep], obs_z[keep]
    R, t, X, reproj, dres = bundle_adjust(list(R), list(t), cams, obs_cam, obs_pt, obs_uv, obs_z, X[used], ref,
                                          args.depth_sigma, args.max_nfev, print)
    print(f"  round 2: median reprojection {np.median(reproj):.2f} px, p90 {np.percentile(reproj, 90):.2f} px, "
          f"median |depth residual| {np.nanmedian(np.abs(dres)) * 1000:.1f} mm")

    # ---- world frame ------------------------------------------------------------------------
    centers = np.array([-R[c].T @ t[c] for c in range(n_cam)])
    axes = np.array([R[c].T @ np.array([0, 0, 1.0]) for c in range(n_cam)])
    up_hint = -np.mean([R[c].T @ np.array([0, 1.0, 0]) for c in range(n_cam)], axis=0)  # image up = -y
    up_hint /= np.linalg.norm(up_hint)
    mid = frames[len(frames) // 2]
    fused_pts, fused_rgb, fused_cam = [], [], []
    for ci, cam in enumerate(cams):
        dep = cv2.imread(str(data / "depth" / cam.folder / f"{mid:06d}.png"), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(str(data / "images" / cam.folder / f"{mid:06d}.{args.image_ext}"))
        _, Xc = depth_in_color(cam, dep)
        u = np.round(cam.K[0, 0] * Xc[:, 0] / Xc[:, 2] + cam.K[0, 2]).astype(int)
        v = np.round(cam.K[1, 1] * Xc[:, 1] / Xc[:, 2] + cam.K[1, 2]).astype(int)
        ok = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        Xw = (Xc[ok] - t[ci]) @ R[ci]
        fused_pts.append(Xw)
        fused_rgb.append(img[v[ok], u[ok]][:, ::-1])
        fused_cam.append(np.full(len(Xw), ci))
    fused_pts = np.concatenate(fused_pts)
    fused_rgb = np.concatenate(fused_rgb)
    fused_cam = np.concatenate(fused_cam)

    world_T = np.eye(4)
    floor_info = None
    look = look_at_point(centers, axes)
    if args.world_frame == "floor":
        sub = fused_pts[np.random.default_rng(1).choice(len(fused_pts), min(200000, len(fused_pts)), replace=False)]
        fl = fit_floor(sub, up_hint, centers)
        if fl is None:
            print("WARNING: floor plane not found; keeping reference-camera frame")
        else:
            nrm, d, support = fl
            y = nrm
            # x: horizontal component of the reference camera's x axis
            xr = R[ref].T @ np.array([1.0, 0, 0])
            x = xr - (xr @ y) * y
            x /= np.linalg.norm(x)
            z = np.cross(x, y)
            Rw = np.stack([x, y, z])  # rows: world axes in old coords
            foot = look - (look @ nrm + d) * nrm  # look-at point projected onto floor
            world_T[:3, :3] = Rw
            world_T[:3, 3] = -Rw @ foot
            floor_info = {"normal_old_frame": nrm.tolist(), "support_points": int(support),
                          "camera_heights_m": (centers @ nrm + d).tolist()}
    Rw, tw = world_T[:3, :3], world_T[:3, 3]
    # X_new = Rw X_old + tw  ->  camera: x = R X_old + t = R Rw^T (X_new - tw) + t
    R_new = [R[c] @ Rw.T for c in range(n_cam)]
    t_new = [t[c] - R[c] @ Rw.T @ tw for c in range(n_cam)]
    X_new = X @ Rw.T + tw
    fused_new = fused_pts @ Rw.T + tw
    centers_new = np.array([-R_new[c].T @ t_new[c] for c in range(n_cam)])
    look_new = Rw @ look + tw

    # ---- QA: cross-camera depth agreement ----------------------------------------------------
    zmaps = []
    for ci, cam in enumerate(cams):
        dep = cv2.imread(str(data / "depth" / cam.folder / f"{mid:06d}.png"), cv2.IMREAD_UNCHANGED)
        zmaps.append(depth_in_color(cam, dep)[0])
    agree = {}
    tiles = []
    for ci, cam in enumerate(cams):
        others = fused_new[fused_cam != ci]
        Xc = others @ R_new[ci].T + t_new[ci]
        okz = Xc[:, 2] > 0.1
        Xc = Xc[okz]
        u = np.round(cam.K[0, 0] * Xc[:, 0] / Xc[:, 2] + cam.K[0, 2]).astype(int)
        v = np.round(cam.K[1, 1] * Xc[:, 1] / Xc[:, 2] + cam.K[1, 2]).astype(int)
        ok = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        zo = Xc[ok, 2]
        zm = zmaps[ci][v[ok], u[ok]]
        valid = np.isfinite(zm)
        rel = np.abs(zo[valid] - zm[valid])
        front = rel[rel < 0.15]  # visible surface (ignore occluded)
        agree[cam.folder] = {"median_abs_mm": float(np.median(front) * 1000) if len(front) else None,
                             "p90_abs_mm": float(np.percentile(front, 90) * 1000) if len(front) else None,
                             "overlap_fraction": float(len(front) / max(len(zo), 1))}
        img = cv2.imread(str(data / "images" / cam.folder / f"{mid:06d}.{args.image_ext}"))
        splat = img.copy()
        oc = fused_cam[fused_cam != ci][okz][ok]
        palette = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255], [255, 0, 255], [255, 255, 0],
                            [128, 128, 255], [128, 255, 128]], dtype=np.uint8)
        sel = np.random.default_rng(2).random(ok.sum()) < 0.25
        splat[v[ok][sel], u[ok][sel]] = palette[oc[sel] % len(palette)]
        cv2.putText(splat, cam.folder, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        tiles.append(cv2.resize(splat, (cam.width // 3, cam.height // 3)))
    h = max(t_.shape[0] for t_ in tiles)
    tiles = [cv2.copyMakeBorder(t_, 0, h - t_.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t_ in tiles]
    cv2.imwrite(str(report_dir / "calibration_reprojection.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # ---- write PLY ----------------------------------------------------------------------------
    ply = report_dir / "calibration_fused.ply"
    sel = np.random.default_rng(3).choice(len(fused_new), min(1_500_000, len(fused_new)), replace=False)
    with open(ply, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(sel)}\nproperty float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(sel), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = fused_new[sel].T
        rec["r"], rec["g"], rec["b"] = fused_rgb[sel].T
        fh.write(rec.tobytes())

    # ---- per-camera stats ------------------------------------------------------------------------
    per_cam = {}
    for ci, cam in enumerate(cams):
        m = obs_cam == ci
        per_cam[cam.folder] = {
            "source": cam.source,
            "observations": int(m.sum()),
            "reproj_median_px": float(np.median(reproj[m])) if m.any() else None,
            "reproj_p90_px": float(np.percentile(reproj[m], 90)) if m.any() else None,
            "depth_residual_median_mm": float(np.nanmedian(np.abs(dres[m])) * 1000) if np.isfinite(dres[m]).any() else None,
            "center_world_m": centers_new[ci].tolist(),
            "depth_agreement_vs_other_cameras": agree[cam.folder],
        }

    # ---- write COLMAP + reference_export -----------------------------------------------------------
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    cam_lines = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    img_lines = ["# Image list with two lines of data per image:",
                 "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# POINTS2D[] as (X, Y, POINT3D_ID)"]
    ref_cams = []
    for i, cam in enumerate(cams, start=1):
        Rc, tc = R_new[i - 1], t_new[i - 1]
        params = [cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]]
        name = f"{cam.folder}/000001.{args.image_ext}"
        cam_lines.append(f"{i} PINHOLE {cam.width} {cam.height} " + " ".join(f"{p:.17g}" for p in params))
        img_lines.append(f"{i} " + " ".join(f"{v:.17g}" for v in (*rotation_to_qvec(Rc), *tc)) + f" {i} {name}")
        img_lines.append("")
        ref_cams.append({"name": name, "folder": cam.folder, "image_name": f"000001.{args.image_ext}",
                         "source_camera_id": cam.source, "R": Rc.tolist(), "t": tc.tolist(),
                         "camera_model_name": "PINHOLE", "width": cam.width, "height": cam.height, "params": params})
    (sparse / "cameras.txt").write_text("\n".join(cam_lines) + "\n", encoding="utf-8")
    (sparse / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")
    (sparse / "points3D.txt").write_text("# 3D point list with one line of data per point:\n"
                                         "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n", encoding="utf-8")
    radii = np.linalg.norm(centers_new - look_new, axis=1)
    scene_meta = {
        "scene_center": look_new.tolist(),
        "scene_radius_p90": float(np.percentile(radii, 90)),
        "scene_radius_p95": float(np.percentile(radii, 95)),
        "scene_radius_max": float(radii.max()),
        "n_points": int(len(X_new)),
        "bounds_source": "azure_kinect_camera_ring",
        "calibration_scale": "metres (Azure Kinect depth)",
        "up_axis": "+Y" if args.world_frame == "floor" and floor_info else "camera",
    }
    (data / "sparse" / "scene_meta.json").write_text(json.dumps(scene_meta, indent=2) + "\n", encoding="utf-8")
    (data / "reference_export.json").write_text(json.dumps({
        "cameras": ref_cams, "scene_meta": scene_meta, "calibration_source": "azure_kinect_depth_sift_ba",
    }, indent=2) + "\n", encoding="utf-8")
    np.save(report_dir / "calibration_points.npy", X_new.astype(np.float32))

    report = {
        "sample_frames": frames.tolist(),
        "pairs": pair_info,
        "points": int(len(X_new)),
        "observations": int(len(obs_cam)),
        "reprojection_median_px": float(np.median(reproj)),
        "reprojection_p90_px": float(np.percentile(reproj, 90)),
        "cameras": per_cam,
        "floor": floor_info,
        "scene_meta": scene_meta,
    }
    (report_dir / "calibration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nPer camera:")
    for f, s in per_cam.items():
        a = s["depth_agreement_vs_other_cameras"]
        print(f"  {f} ({s['source']}): reproj median {s['reproj_median_px']:.2f} px p90 {s['reproj_p90_px']:.2f} px, "
              f"depth vs others median {a['median_abs_mm']:.1f} mm, centre {np.round(s['center_world_m'], 2).tolist()}")
    if floor_info:
        print(f"Camera heights above floor (m): {np.round(floor_info['camera_heights_m'], 2).tolist()}")
    print(f"Wrote {sparse}, {data / 'reference_export.json'}, {report_dir / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
