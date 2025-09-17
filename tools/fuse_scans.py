#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fügt mehrere scan_* Ordner (kuka_lab) per ICP zu einer farbigen Gesamt-Punktwolke zusammen.
Erwartete Struktur:
  kuka_lab/
    index.txt           # Mapping: bag.mcap: scan_001 – scan_004
    scan_001/
      pointcloud.npy    # Nx3 oder strukturiert mit Feldern x,y,z
      image.png         # RGB-Bild der Monokamera
    scan_002/
      ...
Ausgabe:
  kuka_lab_fused/
    fused_map_colored.ply
    transformations.json

Abhängigkeiten: open3d, numpy, opencv-python
Install:
  pip install open3d opencv-python numpy

# Beispiel:
python fuse_scans.py -s "/Pfad/zu/kuka_lab"

# Optional mit eigener Kalibrierung (JSON mit Feldern K, R_lidar_to_cam, t_lidar_to_cam):
python fuse_scans.py -s "/Pfad/zu/kuka_lab" --calib_json calib.json

# Ohne Farbzuweisung (nur Geometrie):
python fuse_scans.py -s "/Pfad/zu/kuka_lab" --no_color

# Parameter für ICP und Downsampling anpassen:
python fuse_scans.py -s "/Pfad/zu/kuka_lab" --voxel 0.015 --thresh 0.08

"""

from __future__ import annotations
import os, re, json, argparse, copy
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import open3d as o3d  # type: ignore

# --------------------------- Default Calibration -----------------------------
# Intrinsics (anpassbar). Default aus deinem Projekt.

K = np.array([
    [679.119,   0.0, 409.699],
    [  0.0,   677.096, 313.462],
    [  0.0,     0.0,     1.0 ],
], dtype=np.float64)

# Extrinsics LiDAR -> Kamera (4x4), Default aus deinem Projekt.
R_lidar_to_cam = np.array([
    [ 0.038044,  -0.999249,  -0.007335],
    [-0.017242,   0.006682,  -0.999829],
    [ 0.999127,   0.038164,  -0.016974],
], dtype=np.float64)
t_lidar_to_cam = np.array([-0.083912,  -0.101645,   0.012768], dtype=np.float64)
T_lidar_to_cam = np.eye(4, dtype=np.float64)
T_lidar_to_cam[:3, :3] = R_lidar_to_cam
T_lidar_to_cam[:3, 3]  = t_lidar_to_cam

# ------------------------------- Utilities -----------------------------------
def load_calib_json(path: Path) -> None:
    """Optional: überschreibt K, R_lidar_to_cam, t_lidar_to_cam aus JSON."""
    global K, R_lidar_to_cam, t_lidar_to_cam, T_lidar_to_cam
    with open(path, "r") as f:
        data = json.load(f)
    if "K" in data: K = np.array(data["K"], dtype=np.float64)
    if "R_lidar_to_cam" in data:
        R = np.array(data["R_lidar_to_cam"], dtype=np.float64)
        R_lidar_to_cam[:] = R
    if "t_lidar_to_cam" in data:
        t = np.array(data["t_lidar_to_cam"], dtype=np.float64).ravel()
        t_lidar_to_cam[:] = t
    T_lidar_to_cam = np.eye(4, dtype=np.float64)
    T_lidar_to_cam[:3,:3] = R_lidar_to_cam
    T_lidar_to_cam[:3, 3] = t_lidar_to_cam

def load_pointcloud_npy(path: Path) -> np.ndarray:
    """
    Lädt .npy mit Nx3 float oder strukturiert {x,y,z}.
    Gibt (N,3) float64 zurück.
    """
    pc = np.load(str(path))
    if getattr(pc, "dtype", None) is not None and pc.dtype.names:
        pc = np.stack([pc["x"], pc["y"], pc["z"]], axis=-1)
    pc = np.asarray(pc, dtype=np.float64)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"Unexpected pointcloud shape in {path}: {pc.shape}")
    return pc[:, :3]

def ndarray_to_o3d(pc: np.ndarray) -> o3d.geometry.PointCloud:
    """(N,3) -> o3d PointCloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.astype(np.float64))
    return pcd

def parse_mapping_file(mapping_file: Path) -> List[Tuple[str, int, int]]:
    """
    Liest index.txt-Zeilen der Form:
      some_bag.mcap: scan_001 – scan_004
    und gibt [(bag, start, end), ...] zurück.
    """
    mapping: List[Tuple[str,int,int]] = []
    with open(mapping_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            m = re.match(r"(.+?):\s*scan_(\d+)\s*[–-]\s*scan_(\d+)", s)
            if not m:
                raise ValueError(f"Konnte Zeile nicht parsen: {s}")
            bag, start, end = m.groups()
            mapping.append((bag.strip(), int(start), int(end)))
    return mapping

def discover_scans_by_mapping(base: Path, mapping_file: Path) -> List[Path]:
    """
    Liefert eine geordnete Liste der scan_* Ordner gemäß index.txt.
    """
    result: List[Path] = []
    mapping = parse_mapping_file(mapping_file)
    for _, s, e in mapping:
        for i in range(s, e+1):
            p = base / f"scan_{i:03d}"
            if p.is_dir():
                result.append(p)
    # Falls index.txt weniger Scans auflistet als vorhanden, Rest anhängen (sortiert)
    existing = set(p.name for p in result)
    extra = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("scan_") and p.name not in existing],
                   key=lambda x: int(x.name.split("_")[-1]))
    result.extend(extra)
    return result

# ----------------------------- ICP + Colorize --------------------------------
def add_scan_to_map(new_scan: o3d.geometry.PointCloud,
                    global_map: o3d.geometry.PointCloud | None,
                    threshold: float = 0.05,
                    voxel_size: float = 0.01) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Inkrementelles ICP: richtet new_scan an global_map aus und fügt ihn hinzu.
    Rückgabe: (aktualisierte_map, T_map_from_scan)
    """
    # erster Scan -> Map initialisieren
    if (global_map is None) or (len(global_map.points) == 0):
        gm = o3d.geometry.PointCloud()
        gm.points = copy.deepcopy(new_scan.points)
        if new_scan.has_colors():
            gm.colors = copy.deepcopy(new_scan.colors)
        return gm, np.eye(4)

    # Map für ICP etwas ausdünnen (Stabilität)
    map_down = global_map.voxel_down_sample(voxel_size)

    trans_init = np.eye(4)
    reg = o3d.pipelines.registration.registration_icp(
        new_scan, map_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    T = reg.transformation  # T_map_from_scan

    new_aligned = copy.deepcopy(new_scan)
    new_aligned.transform(T)

    global_map += new_aligned
    global_map = global_map.voxel_down_sample(voxel_size)
    return global_map, T

def swap_rb(rgb: np.ndarray) -> np.ndarray:
    """R<->B tauschen (für Dateien, die fälschlich als BGR gespeichert wurden)."""
    return rgb[..., ::-1]

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
    if len(scan_map.points) == 0: return
    H, W = img_bgr.shape[:2]
    fx, fy, cx, cy = K_[0, 0], K_[1, 1], K_[0, 2], K_[1, 2]

    P = np.asarray(scan_map.points, dtype=np.float64)
    ones = np.ones((P.shape[0], 1), dtype=np.float64)
    P_h = np.hstack([P, ones])       # (N,4)

    Pc = (T_cam_from_map @ P_h.T).T  # (N,4) -> Kamera-KS
    Xc, Yc, Zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    in_front = Zc > 0.0
    if not np.any(in_front): return

    u = (fx * (Xc / Zc) + cx)
    v = (fy * (Yc / Zc) + cy)

    if bilinear:
        valid = (in_front & (u >= 0) & (v >= 0) & (u < (W - 1)) & (v < (H - 1)))
    else:
        u_i = np.round(u).astype(np.int32)
        v_i = np.round(v).astype(np.int32)
        valid = (in_front & (u_i >= 0) & (v_i >= 0) & (u_i < W) & (v_i < H))
    #img_rgb = swap_rb(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    colors = np.zeros((P.shape[0], 3), dtype=np.float64)  # immer frisch

    if bilinear:
        u0 = np.floor(u[valid]).astype(np.int32)
        v0 = np.floor(v[valid]).astype(np.int32)
        du = u[valid] - u0
        dv = v[valid] - v0
        c00 = img_rgb[v0,     u0,     :]
        c10 = img_rgb[v0,     u0 + 1, :]
        c01 = img_rgb[v0 + 1, u0,     :]
        c11 = img_rgb[v0 + 1, u0 + 1, :]
        c = ((1 - du)[:, None] * (1 - dv)[:, None] * c00 +
             (    du)[:, None] * (1 - dv)[:, None] * c10 +
             (1 - du)[:, None] * (    dv)[:, None] * c01 +
             (    du)[:, None] * (    dv)[:, None] * c11)
        colors[valid] = c
    else:
        colors[valid] = img_rgb[v_i[valid], u_i[valid], :]

    scan_map.colors = o3d.utility.Vector3dVector(colors)

# ------------------------------- Main logic ----------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Füge scan_* (kuka_lab) per ICP zu farbiger Gesamt-Punktwolke zusammen.")
    ap.add_argument("-s", "--source", required=True, help="Pfad zum Ordner 'kuka_lab'")
    ap.add_argument("--mapping", default="index.txt", help="Dateiname der Mapping-Datei (Default: index.txt)")
    ap.add_argument("-o", "--out", default="", help="Ausgabeordner (Default: <source>_fused)")
    ap.add_argument("--voxel", type=float, default=0.01, help="Voxelgröße fürs Downsampling [m] (Default: 0.01)")
    ap.add_argument("--thresh", type=float, default=0.05, help="ICP max correspondence distance [m] (Default: 0.05)")
    ap.add_argument("--calib_json", default="", help="Optional: Kalibrierung JSON (K, R_lidar_to_cam, t_lidar_to_cam)")
    ap.add_argument("--no_color", action="store_true", help="Nur Geometrie fusionieren (keine Farbzuweisung)")
    ap.add_argument("--show", action="store_true", help="3D-Viewer nach Abschluss anzeigen")
    args = ap.parse_args()

    base = Path(os.path.abspath(os.path.expanduser(args.source)))
    if not base.is_dir():
        raise SystemExit(f"Kein Verzeichnis: {base}")

    mapping_file = base / args.mapping
    if mapping_file.is_file():
        scans = discover_scans_by_mapping(base, mapping_file)
    else:
        scans = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("scan_")],
                       key=lambda p: int(p.name.split("_")[-1]))
    if not scans:
        raise SystemExit(f"Keine scan_* Ordner in {base} gefunden.")

    out_dir = Path(args.out) if args.out else (base.parent / f"{base.name}_fused")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.calib_json:
        load_calib_json(Path(args.calib_json))

    # ICP-Fusion
    scan_map: o3d.geometry.PointCloud | None = None
    Ts: Dict[str, List[List[float]]] = {}  # T_map_from_scan je scan_XXX

    for i, scan_path in enumerate(scans, start=1):
        pc_path = scan_path / "pointcloud.npy"
        img_path = scan_path / "image.png"
        if not pc_path.is_file():
            print(f"[WARN] {scan_path.name}: pointcloud.npy fehlt – übersprungen.")
            continue

        try:
            pts = load_pointcloud_npy(pc_path)
        except Exception as e:
            print(f"[WARN] Konnte {pc_path} nicht laden: {e} – übersprungen.")
            continue

        pcd = ndarray_to_o3d(pts)
        scan_map, T = add_scan_to_map(pcd, scan_map, threshold=args.thresh, voxel_size=args.voxel)
        Ts[scan_path.name] = T.tolist()

        # Farbzuteilung mit aktuellem Bild (wenn vorhanden und erlaubt)
        if not args.no_color and img_path.is_file():
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None and img_bgr.size > 0:
                # Kamera-from-Map: T_cam_from_map = T_lidar_to_cam @ inv(T_map_from_scan)
                T_cam_from_map = T_lidar_to_cam @ np.linalg.inv(T)
                colorize_map_with_image(scan_map, img_bgr, K, T_cam_from_map, bilinear=False)
            else:
                print(f"[WARN] Konnte {img_path} nicht lesen – Scan bleibt ggf. unkoloriert.")
        elif args.no_color:
            pass
        else:
            print(f"[WARN] {scan_path.name}: image.png fehlt – Scan bleibt ggf. unkoloriert.")

        print(f"[OK] Ausgerichtet: {scan_path.name}  | Punkte in Map: {len(scan_map.points)}")

    # Ausgabe
    ply_path = out_dir / "fused_map_colored.ply"
    ok = o3d.io.write_point_cloud(str(ply_path), scan_map, write_ascii=False)
    if not ok:
        raise SystemExit(f"Speichern fehlgeschlagen: {ply_path}")
    print(f"\nFertig. Gespeichert: {ply_path}")

    with open(out_dir / "transformations.json", "w") as f:
        json.dump(Ts, f, indent=2)
    print(f"Posen je Scan: {out_dir/'transformations.json'}")

    if args.show:
        o3d.visualization.draw_geometries([scan_map]) # type: ignore

if __name__ == "__main__":
    main()
