#!/usr/bin/env python3
# fix_dataset_png_rb.py
import argparse, os, glob
from PIL import Image
import numpy as np

def swap_rb(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., ::-1]

def main():
    ap = argparse.ArgumentParser(description="Swap R/B in allen scan_*/image.png")
    ap.add_argument("root", help="Ordner mit scan_*/image.png (z.B. kuka_lab oder kuka_lab_2)")
    ap.add_argument("--inplace", action="store_true", help="Original überschreiben (sonst image_rgb.png schreiben)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "scan_*", "image.png")))
    if not paths:
        print("Keine Bilder gefunden.")
        return
    print(f"Bearbeite {len(paths)} Bilder …")

    for p in paths:
        img = Image.open(p).convert("RGB")
        fixed = swap_rb(np.array(img))
        out = p if args.inplace else os.path.join(os.path.dirname(p), "image_rgb.png")
        Image.fromarray(fixed).save(out)
    print("Fertig.")

if __name__ == "__main__":
    main()
