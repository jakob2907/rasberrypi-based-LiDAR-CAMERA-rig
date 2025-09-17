import argparse
import numpy as np
import open3d as o3d

def main():
    parser = argparse.ArgumentParser(description="Zeige .ply Punktwolke mit Open3D")
    parser.add_argument("ply", help="Pfad zur .ply-Datei")
    parser.add_argument("--no_colors", action="store_true",
                        help="Ignoriere Farbinformationen und zeige alles in Grau")
    parser.add_argument("--voxel", type=float, default=0.0,
                        help="Optionales Voxel-Downsampling (z.B. 0.01)")
    args = parser.parse_args()

    # Punktwolke laden
    pcd = o3d.io.read_point_cloud(args.ply)
    if not pcd.has_points():
        raise SystemExit("Fehler: Keine Punkte in der .ply-Datei gefunden.")

    # Optionales Downsampling
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)

    # Falls keine Farben vorhanden oder --no_colors angegeben
    if args.no_colors or not pcd.has_colors():
        n = np.asarray(pcd.points).shape[0]
        gray = np.ones((n, 3)) * 0.5
        pcd.colors = o3d.utility.Vector3dVector(gray)

    # Punktwolke anzeigen
    o3d.visualization.draw_geometries([pcd])  # type: ignore

if __name__ == "__main__":
    main()
