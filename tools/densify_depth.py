#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth image hole filling as a single callable function.

Example:
    filled = fill_depth_image(
        "depth_0001.png",
        method="nearest+bilateral",
        bilateral_d=7,
        bilateral_sigma_color=50.0,
        bilateral_sigma_space=5.0,
        save_to="depth_0001_filled.png"  # optional
    )
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import numpy as np
import cv2

ArrayOrPath = Union[np.ndarray, str, Path]

# ------------------------ helpers ------------------------------------------
def _as_depth_array(inp: ArrayOrPath) -> np.ndarray:
    """Accept a path (str/Path) or a numpy array and return a single-channel array."""
    if isinstance(inp, (str, Path)):
        img = cv2.imread(str(inp), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {inp}")
    elif isinstance(inp, np.ndarray):
        img = inp
    else:
        raise TypeError("png_or_array must be a numpy array or a str/Path.")

    if img.ndim == 3:
        # Convert accidental 3-channel depth PNGs to single channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def _valid_mask(depth: np.ndarray) -> np.ndarray:
    return (np.isfinite(depth) & (depth > 0)) if np.issubdtype(depth.dtype, np.floating) else (depth > 0)

def _edt_nearest_fill(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    holes = ~valid
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore
        idx = distance_transform_edt(holes, return_distances=False, return_indices=True)
        filled = depth.copy()
        filled[holes] = depth[tuple(ind[holes] for ind in idx)] # type: ignore
        return filled
    except Exception:
        filled = depth.copy()
        kernel = np.ones((3, 3), np.uint8)
        # conservative number of iterations; stops early if no holes remain
        for _ in range(64):
            mask_holes = (filled == 0) if not np.issubdtype(filled.dtype, np.floating) \
                         else (~np.isfinite(filled)) | (filled <= 0)
            if not mask_holes.any():
                break
            dil = cv2.dilate(filled, kernel, iterations=1)
            take = mask_holes & (dil > 0)
            filled[take] = dil[take]
        return filled

def _griddata_fill(depth: np.ndarray, valid: np.ndarray, mode: str) -> np.ndarray:
    try:
        from scipy.interpolate import griddata  # type: ignore
    except Exception:
        return _edt_nearest_fill(depth, valid)

    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack((yy[valid], xx[valid]))
    vals = depth[valid].astype(np.float32)
    target = np.column_stack((yy.ravel(), xx.ravel()))
    interp = griddata(pts, vals, target, method=mode, fill_value=np.nan).reshape(h, w)

    if np.isnan(interp).any():
        nn = griddata(pts, vals, target, method="nearest").reshape(h, w)
        m = np.isnan(interp)
        interp[m] = nn[m]
    return interp.astype(depth.dtype, copy=False)

def _bilateral_refine(arr: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    orig_dtype = arr.dtype
    arr32 = arr.astype(np.float32)
    out = cv2.bilateralFilter(arr32, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        out = np.clip(out, info.min, info.max)
    return out.astype(orig_dtype)

# ------------------------ public API ----------------------------------------
def fill_depth_image(
    png_or_array: ArrayOrPath,
    *,
    method: str = "nearest+bilateral",         # "nearest" | "nearest+bilateral" | "linear" | "cubic"
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 50.0,       # in depth units (e.g., millimeters)
    bilateral_sigma_space: float = 5.0,        # in pixels
    save_to: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Fill holes in a sparse depth image (0/NaN treated as holes).

    png_or_array : str | Path | np.ndarray
        Either a path to a depth PNG or an already loaded single-channel array.
    """
    depth = _as_depth_array(png_or_array)
    valid = _valid_mask(depth)
    if valid.sum() == 0:
        raise RuntimeError("No valid (non-zero) pixels detected in input depth.")

    if method == "nearest":
        filled = _edt_nearest_fill(depth, valid)
    elif method == "nearest+bilateral":
        filled = _edt_nearest_fill(depth, valid)
        filled = _bilateral_refine(filled, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    elif method in ("linear", "cubic"):
        filled = _griddata_fill(depth, valid, mode=method)
    else:
        raise ValueError('Unknown method. Use "nearest", "nearest+bilateral", "linear", or "cubic".')

    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_to), filled):
            raise IOError(f"Failed to write output image: {save_to}")
    return filled

# ------------------------ optional CLI --------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fill holes in a sparse depth image from file or array.")
    ap.add_argument("--in", dest="inp", required=True, help="Input depth PNG")
    ap.add_argument("--out", dest="outp", help="Optional output path")
    ap.add_argument("--method", default="nearest+bilateral",
                    choices=["nearest", "nearest+bilateral", "linear", "cubic"])
    ap.add_argument("--bilateral_d", type=int, default=7)
    ap.add_argument("--bilateral_sigma_color", type=float, default=50.0)
    ap.add_argument("--bilateral_sigma_space", type=float, default=5.0)
    args = ap.parse_args()

    result = fill_depth_image(
        args.inp, method=args.method,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        save_to=args.outp
    )
    print(f"Done. Result shape={result.shape}, dtype={result.dtype}")
