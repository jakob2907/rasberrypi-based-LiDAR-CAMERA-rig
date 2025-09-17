#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UniFusion output generator for scan_* folders.

For each scan_XXX directory containing:
  - image.png
  - pointcloud.npy  (Nx3 float or structured with fields x,y,z)

This script writes into:
  fusion_output_<source_name>/
    frame0001.png  (RGB)
    depth0001.png  (uint8 depth visualization)

Options:
  --mode_depth 0 : depth at image resolution, nearest-depth per pixel (default)
  --mode_depth 1 : depth scaled to (NEW_H, NEW_W)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import open3d as o3d
from icp import add_scan_to_map
from densify_depth import fill_depth_image

# ----------------------------- Calibration -----------------------------------

# Intrinsics (adjust if needed)
K = np.array(
    [
        [968.691, 0.0, 309.268],
        [0.0, 965.205, 231.423],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

K_07_09 = np.array(
    [
        [1215.32, 0.0, 1047.87],
        [0.0, 1214.73, 745.068],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

K_global_shutter = np.array(
    [
        [679.119, 0.0, 409.699],
        [0.0, 677.096, 313.462],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# LiDAR -> Camera extrinsics
R_lidar_to_cam = np.array(
    [
        [0.194503, -0.980898, 0.002583],
        [-0.120718, -0.026551, -0.992332],
        [0.973445,  0.192700, -0.123576],
    ],
    dtype=np.float64,
)
t_lidar_to_cam = np.array([-0.139002, 0.155894, -0.392262], dtype=np.float64)


T_lidar_to_cam = np.array(
    [
        [0.194503, -0.980898, 0.002583, -0.139002],
        [-0.120718, -0.026551, -0.992332, 0.155894],
        [0.973445,  0.192700, -0.123576, -0.392262],
        [0, 0,  0,  1]
    ],
    dtype=np.float64,
)

T_lidar_to_cam_07_09 = np.array(
    [
        [0.499221,  -0.864892,  -0.052346, -2.895656],
        [0.234355,   0.076617,   0.969127, -1.877747],
        [-0.834180,  -0.496076,   0.240941, 5.310832],
        [0, 0,  0,  1]
    ],
    dtype=np.float64,
)

T_lidar_to_cam_global_shutter = np.array(
    [
        [0.038044,  -0.999249,  -0.007335, -0.083912],
        [-0.017242,   0.006682,  -0.999829, -0.101645],
        [0.999127,   0.038164,  -0.016974, 0.012768],
        [0, 0,  0,  1]
    ],
    dtype=np.float64,
)
# Target size for scaled depth (mode_depth=1)
NEW_W = 400
NEW_H = 300


# ------------------------------- Utilities -----------------------------------

def filter_scan_map_to_view(
    scan_map: o3d.geometry.PointCloud,
    K_: np.ndarray,
    T_cam_from_map: np.ndarray,
    img_shape,              # (H, W, 3)
    z_min: float = 0.0,
    z_max: float | None = None, # type: ignore
    margin: int = 0         # optionaler Pixelrand
) -> o3d.geometry.PointCloud:
    """
    Gibt eine gefilterte Kopie der scan_map zurück, die nur Punkte enthält,
    die vor der aktuellen Kamera liegen (Zc > z_min) und im (erweiterten) Bildfeld sind.
    """
    if len(scan_map.points) == 0:
        return scan_map

    H, W = img_shape[:2]
    fx, fy, cx, cy = K_[0, 0], K_[1, 1], K_[0, 2], K_[1, 2]

    P = np.asarray(scan_map.points, dtype=np.float64)            # (N,3)
    ones = np.ones((P.shape[0], 1), dtype=np.float64)
    P_h = np.hstack([P, ones])                                   # (N,4)
    Pc = (T_cam_from_map @ P_h.T).T                              # (N,4)
    Xc, Yc, Zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]

    # vor der Kamera
    in_front = Zc > max(1e-6, z_min)
    if z_max is not None:
        in_front &= (Zc < z_max)

    # nur gültige Z für Pixelprojektion benutzen
    Z_safe = np.clip(Zc, 1e-6, None)
    u = fx * (Xc / Z_safe) + cx
    v = fy * (Yc / Z_safe) + cy

    # im Bild (+ optionaler Margin)
    in_img = (u >= -margin) & (u < W + margin) & (v >= -margin) & (v < H + margin)

    keep = np.nonzero(in_front & in_img)[0]
    if keep.size == 0:
        # keine sichtbaren Punkte – leere Kopie zurückgeben
        return o3d.geometry.PointCloud()

    return scan_map.select_by_index(keep)

def get_source_name(source_path: str) -> str:
    """Derive a friendly source name from the given path."""
    sp = os.path.expanduser(source_path)
    base = os.path.basename(sp.rstrip("/"))
    return base.replace("synced_data_", "")

def get_fusion_output_dir(source_dir: str | Path) -> str: # type: ignore
    """
    Returns a sibling directory '<basename(source_dir)>_fusion' and creates it.
    Example: -s data/home_28_08  ->  data/home_28_08_fusion
    """
    src = Path(source_dir).resolve()
    out = src.parent / f"{src.name}_fusion"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)

def load_pointcloud(path: str) -> np.ndarray:
    """
    Load point cloud from .npy (Nx3 float or structured with x,y,z).
    Returns (N,3) float32.
    """
    pc = np.load(path)
    if pc.dtype.names:  # structured array
        pc = np.stack([pc["x"], pc["y"], pc["z"]], axis=-1)
    return pc.astype(np.float32, copy=False)

def project_points_with_depth(
    points: np.ndarray,
    K_: np.ndarray,
    T_cam_from_lidar: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projiziere 3D-LiDAR-Punkte in die Kameraebene.

    Args:
        points: (N,3) numpy array mit LiDAR Punkten.
        K_: (3,3) Kamera- Intrinsic Matrix.
        T_cam_from_lidar: (4,4) Transformation LiDAR->Kamera.

    Returns:
        uv: (M,2) int Pixel-Koordinaten innerhalb des Bildes.
        depths: (M,) float Tiefenwerte in Metern.
    """
    if points.shape[1] != 3:
        raise ValueError("points muss (N,3) sein")

    # Homogene Koordinaten (N,4)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    # LiDAR → Kamera transformieren
    pts_cam = (T_cam_from_lidar @ pts_h.T).T
    Xc, Yc, Zc = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # nur Punkte vor der Kamera
    mask = Zc > 0
    Xc, Yc, Zc = Xc[mask], Yc[mask], Zc[mask]

    if Xc.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    # Projektion in Pixel
    fx, fy, cx, cy = K_[0, 0], K_[1, 1], K_[0, 2], K_[1, 2]
    u = (fx * (Xc / Zc) + cx).astype(np.int32)
    v = (fy * (Yc / Zc) + cy).astype(np.int32)

    uv = np.stack([u, v], axis=1)
    depths = Zc.astype(np.float32)

    return uv, depths

def depth_map_unscaled(
    image_shape: Tuple[int, int, int], uv: np.ndarray, depths: np.ndarray
) -> np.ndarray:
    """
    Build depth map (H,W) float32 at image resolution.
    If multiple points hit a pixel, keep nearest depth.
    """
    H, W = image_shape[:2]
    depth = np.zeros((H, W), dtype=np.float32)
    for (u, v), d in zip(uv, depths):
        if 0 <= u < W and 0 <= v < H:
            if depth[v, u] == 0 or d < depth[v, u]:
                depth[v, u] = d
    return depth


def depth_map_scaled(
    image_shape: Tuple[int, int, int], uv: np.ndarray, depths: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Depth map visualization (uint8) at target_shape (H_new, W_new).
    Pixels without depth remain 0.
    """
    H, W = image_shape[:2]
    H_new, W_new = target_shape
    sx = W_new / W
    sy = H_new / H

    uv_s = uv.copy()
    uv_s[:, 0] = (uv[:, 0] * sx).astype(int)
    uv_s[:, 1] = (uv[:, 1] * sy).astype(int)

    dm = np.full((H_new, W_new), np.nan, dtype=np.float32)
    for (u, v), d in zip(uv_s, depths):
        if 0 <= u < W_new and 0 <= v < H_new:
            dm[v, u] = d

    if np.all(np.isnan(dm)):
        return np.zeros((H_new, W_new), dtype=np.uint8)

    dmin = np.nanmin(dm)
    dmax = np.nanmax(dm)
    vis = (dmax - dm) / (dmax - dmin + 1e-6)
    vis = (vis * 255).astype(np.uint8)
    vis[np.isnan(dm)] = 0
    return vis


def depth_to_uint8_vis(dm: np.ndarray) -> np.ndarray:
    """
    Convert a float32 depth map (zeros = no data) into an 8-bit visualization.
    Normalizes over non-zero pixels only; nearer = brighter.
    """
    mask = dm > 0
    if not np.any(mask):
        return np.zeros_like(dm, dtype=np.uint8)
    vis = np.zeros_like(dm, dtype=np.uint8)
    d = dm[mask]
    dmin = float(d.min())
    dmax = float(d.max())
    # invert for visualization (near bright)
    vis_vals = (dmax - d) / (dmax - dmin + 1e-6) * 255.0
    vis[mask] = vis_vals.astype(np.uint8)
    return vis

def ndarray_to_o3d(pc: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Wandelt ein numpy-Array (structured oder normal) in eine Open3D-PointCloud um.

    Unterstützte Eingaben:
      - Structured array mit Feldern 'x','y','z' (+ optional 'r','g','b')
      - Normales Array der Form (N,3) oder (N,6) (x,y,z[,r,g,b])

    Returns:
      o3d.geometry.PointCloud
    """
    pcd = o3d.geometry.PointCloud()

    # Structured array
    if pc.dtype.names:
        # Pflicht: x,y,z
        xyz = np.stack([pc['x'], pc['y'], pc['z']], axis=-1).astype(np.float64)
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Optional: Farben
        if all(name in pc.dtype.names for name in ('r','g','b')):
            rgb = np.stack([pc['r'], pc['g'], pc['b']], axis=-1).astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    # Normales ndarray
    else:
        if pc.shape[1] >= 3:
            xyz = pc[:, :3].astype(np.float64)
            pcd.points = o3d.utility.Vector3dVector(xyz)
        if pc.shape[1] >= 6:
            rgb = pc[:, 3:6].astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    return pcd

def colorize_map_with_image(
    scan_map: o3d.geometry.PointCloud,
    img_bgr: np.ndarray,
    K_: np.ndarray,
    T_cam_from_map: np.ndarray,
    bilinear: bool = False,
) -> None:
    """
    Weist sichtbaren Map-Punkten Farben aus dem aktuellen Kamerabild zu.
    Schreibt direkt in scan_map.colors (RGB in [0,1]).
    """
    if len(scan_map.points) == 0:
        return

    H, W = img_bgr.shape[:2]
    fx, fy, cx, cy = K_[0, 0], K_[1, 1], K_[0, 2], K_[1, 2]

    # Punkte (N,3) -> (N,4)
    P = np.asarray(scan_map.points, dtype=np.float64)
    ones = np.ones((P.shape[0], 1), dtype=np.float64)
    P_h = np.hstack([P, ones])

    # In Kamera-KS transformieren und projizieren
    Pc = (T_cam_from_map @ P_h.T).T
    Xc, Yc, Zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    in_front = Zc > 0.0
    if not np.any(in_front):
        return

    # Pixelkoordinaten
    u = (fx * (Xc / Zc) + cx)
    v = (fy * (Yc / Zc) + cy)

    # gültige Pixel im Bild
    if bilinear:
        # für bilineares Sampling brauchen wir Rand-1 (weil wir u1=v1 verwenden)
        valid = (in_front &
                 (u >= 0) & (v >= 0) &
                 (u < (W - 1)) & (v < (H - 1)))
    else:
        u_i = np.round(u).astype(np.int32)
        v_i = np.round(v).astype(np.int32)
        valid = (in_front &
                 (u_i >= 0) & (v_i >= 0) &
                 (u_i < W) & (v_i < H))

    # BGR -> RGB in [0,1]
    img_rgb = swap_rb(img_bgr)
    #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # existierende Farben oder leeres Array
    if len(scan_map.colors) == len(scan_map.points):
        colors = np.asarray(scan_map.colors, dtype=np.float64)
    else:
        colors = np.zeros((P.shape[0], 3), dtype=np.float64)

    if bilinear:
        # bilineares Sampling (optional)
        u0 = np.floor(u[valid]).astype(np.int32)
        v0 = np.floor(v[valid]).astype(np.int32)
        du = u[valid] - u0
        dv = v[valid] - v0

        # 4 Nachbarn
        c00 = img_rgb[v0,     u0    , :]
        c10 = img_rgb[v0,     u0 + 1, :]
        c01 = img_rgb[v0 + 1, u0    , :]
        c11 = img_rgb[v0 + 1, u0 + 1, :]

        c = ( (1 - du)[:, None] * (1 - dv)[:, None] * c00 +
              (    du)[:, None] * (1 - dv)[:, None] * c10 +
              (1 - du)[:, None] * (    dv)[:, None] * c01 +
              (    du)[:, None] * (    dv)[:, None] * c11 )
        colors[valid] = c
    else:
        # nächster Nachbar
        colors[valid] = img_rgb[v_i[valid], u_i[valid], :]

    scan_map.colors = o3d.utility.Vector3dVector(colors)

def swap_rb(rgb: np.ndarray) -> np.ndarray:
    """R<->B tauschen (für Dateien, die fälschlich als BGR gespeichert wurden)."""
    return rgb[..., ::-1]

# ---------------------------------- Core -------------------------------------

def process_scan(
    scan_dir: Path,
    out_dir: Path,
    frame_idx: int,
    mode_depth: int,
    K_: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    scan_map: o3d.geometry.PointCloud,
    filled: bool,
) -> o3d.geometry.PointCloud:
    """
    Process one scan_XXX folder into (frame####.png, depth####.png) in out_dir.
    Returns True if successful, False otherwise.
    """
    img_path = scan_dir / "image.png"
    pc_path = scan_dir / "pointcloud.npy"

    if not img_path.is_file() or not pc_path.is_file():
        print(f"Skipping {scan_dir.name}: missing image.png or pointcloud.npy")
        return False

    img_bgr = cv2.imread(str(img_path))
    print(img_bgr.shape)
    if img_bgr is None:
        print(f"Failed to read image: {img_path}")
        return False

    try:
        pts = load_pointcloud(str(pc_path))
    except Exception as e:
        print(f"Failed to load point cloud {pc_path}: {e}")
        return False

    '''
        pts from structured ndarray to o3d.geometry.Pointcloud
    '''
    pts = ndarray_to_o3d(pts)
    print(f"Anzahl Punkte in new_scan: {pts}")


    #nur points aden welche vor Kamera liegen



    #added alle points
    scan_map, T_pts = add_scan_to_map(pts, scan_map, threshold=0.05, voxel_size=0.01)

    print(f"Anzahl Punkte in scan_map: {len(scan_map.points)}")
    print(T_pts)

    #T_pts ist Transformation scan_to_map

    '''
        Wenn ich jetzt die ganze Map in den ImageSpace relativ zu dem neuen Scan transformieren will, dann
        muss ich die Transformation anwenden: T = T_lidar_to_cam @ T_pts
    '''

    T_cam_from_map = T_lidar_to_cam_global_shutter @ np.linalg.inv(T_pts)

    # Sichtbare Map (nur Punkte vor der aktuellen Kamera/im Bild)
    visible_map = filter_scan_map_to_view(
        scan_map=scan_map,
        K_=K_global_shutter,
        T_cam_from_map=T_cam_from_map,
        img_shape=img_bgr.shape,
        z_min=0.0,
        z_max=None,     # z.B. 30.0 für 30 m Reichweite
        margin=0
    )
    
    colorize_map_with_image(
        scan_map=visible_map,
        img_bgr=img_bgr,
        K_=K_global_shutter,
        T_cam_from_map=T_cam_from_map,
        bilinear=False
    )

    #scan_map als np array
    scan_map_np = np.asarray(visible_map.points)

    # Project and build depth
    uv, depths = project_points_with_depth(scan_map_np, K_, T_cam_from_map)

    # uv -> alle points in camera_image

    if mode_depth == 1:
        depth_vis = depth_map_scaled(img_bgr.shape, uv, depths, (NEW_H, NEW_W)) # type: ignore
    else:
        dm = depth_map_unscaled(img_bgr.shape, uv, depths) # type: ignore
        depth_vis = depth_to_uint8_vis(dm)

    print(f"filled: {filled}")

    # Densify(fill holes) on the np.array before writing
    if filled is True:
        depth_filled = fill_depth_image(
        depth_vis,                    # pass array directly
        method="nearest+bilateral",   # or: "nearest", "linear", "cubic"
        bilateral_d=7,
        bilateral_sigma_color=50.0,
        bilateral_sigma_space=5.0
        )
    else:
        depth_filled = depth_vis
        

    # Save RGB (convert BGR->RGB for UniFusion)



    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_name = f"frame{frame_idx:04d}.png"
    depth_name = f"depth{frame_idx:04d}.png"

    cv2.imwrite(str(out_dir / rgb_name), rgb)
    cv2.imwrite(str(out_dir / depth_name), depth_filled)

    return scan_map, True

def save_o3d_pcd_as_ply(pcd: o3d.geometry.PointCloud,
                        out_dir: Path,
                        filename: str = "pointcloud.ply",
                        binary: bool = True) -> Path:
    """
    Speichert eine Open3D-PointCloud als .ply in out_dir.
    """
    path = out_dir / (filename if filename.endswith(".ply") else f"{filename}.ply")

    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=not binary)
    if not ok:
        raise IOError(f"Speichern fehlgeschlagen: {path}")
    return path

# ---------------------------------- Main -------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate UniFusion-ready RGB+Depth from scan_* folders")
    ap.add_argument(
        "-s", "--source", required=True,
        help="Path with scan_001, scan_002, ...",
    )
    ap.add_argument(
        "--mode_depth", type=int, choices=[0, 1], default=0,
        help=f"0: native size; 1: scaled to ({NEW_W}x{NEW_H})",
    )
    ap.add_argument(
        "--start_idx", type=int, default=1,
        help="First scan index to process (default: 1 -> scan_001)",
    )
    ap.add_argument(
        "--end_idx", type=int, default=0,
        help="Last scan index to process (0 = auto to last found)",
    )
    ap.add_argument(
    "--filled",
    action="store_true",
    help="Depth Images filled (Bilinear-Interpolation)",
)

    args = ap.parse_args()

    base = Path(os.path.abspath(os.path.expanduser(args.source)))
    if not base.is_dir():
        raise SystemExit(f"Not a directory: {base}")

    # discover scans
    scans: List[Path] = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("scan_")])
    if not scans:
        raise SystemExit(f"No scan_* folders found in {base}")

    # range selection
    def scan_idx(p: Path) -> int:
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return 0

    scans = sorted(scans, key=scan_idx)
    if args.end_idx <= 0:
        end_idx = scan_idx(scans[-1])
    else:
        end_idx = args.end_idx

    start_idx = max(1, args.start_idx)
    scans = [p for p in scans if start_idx <= scan_idx(p) <= end_idx]
    if not scans:
        raise SystemExit(f"No scans in requested range [{start_idx}..{end_idx}]")

    out_dir = Path(get_fusion_output_dir(base))

    frame_counter = 1
    ok_count = 0
    scan_map = o3d.geometry.PointCloud()

    for scan_path in scans:
        scan_map, ok = process_scan(
            scan_dir=scan_path,
            out_dir=out_dir,
            frame_idx=frame_counter,
            mode_depth=args.mode_depth,
            K_=K,
            T_lidar_to_cam=T_lidar_to_cam,
            scan_map=scan_map,
            filled=args.filled
        )
        if ok:
            ok_count += 1
            frame_counter += 1

    #store full pointcloud:
    
    o3d.visualization.draw_geometries([scan_map])  # type: ignore
    path = save_o3d_pcd_as_ply(
        pcd=scan_map,
        out_dir=out_dir,
        filename="scan.ply",
        binary=True)

    print(
        f"Done. Wrote {ok_count} RGB+Depth pairs to: {out_dir} "
        f"(range scan_{start_idx:03d}..scan_{end_idx:03d})"
        f"Pointcloud stored: {path}"
    )


if __name__ == "__main__":
    main()
