"""
pyrsd/core/processing.py
extracts delta displacement from flow and background image by using spline
"""

import numpy as np
from pathlib import Path
from pyrsd.utils.io import (find_images, load_image, image_to_hue_field)

def hue_to_displacement(hue_field: np.ndarray, spline) -> np.ndarray:
    """maps hue value to displacement field value using calibration spline function"""
    disp = np.full(hue_field.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(hue_field)
    if valid.any():
        disp[valid] = spline(hue_field[valid])
    return disp

def plateau_mask(hue_field: np.ndarray, spline, tol_deg: float = 0.5) -> np.ndarray:
    """True where a pixel's hue lies on the calibration's saturation plateau, i.e. its displacement is only
    known to +/- spline.plateau_halfwidth_mm. All False if the calibration has no plateau."""
    ph = getattr(spline, "plateau_hue", None)
    if ph is None:
        return np.zeros(hue_field.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        return np.abs(hue_field - ph) <= tol_deg

def compute_delta_displacement(flow_hue: np.ndarray, bg_hue: np.ndarray, spline) -> np.ndarray:
    """returns delta displacement when flow hue and background hue field are given with calibration spline"""
    return (hue_to_displacement(flow_hue,spline)-hue_to_displacement(bg_hue,spline))

def process_stack(flow_dir: str, bg_path: str, spline, progress_callback=None) -> np.ndarray:
    """takes flow field path, background field path, spline and returns delta displacement"""
    flow_files = find_images(flow_dir)
    if not flow_files:
        raise RuntimeError(f"no flow images found in folder {flow_dir}")
    bg_img = load_image(bg_path)
    bg_hue = image_to_hue_field(bg_img)
    total = len(flow_files)
    delta_field = []

    for i, path in enumerate(flow_files):
        hue = image_to_hue_field(load_image(path))
        delta_field.append(compute_delta_displacement(hue, bg_hue, spline))
        if progress_callback:
            progress_callback(i+1,total, Path(path).name)
    return np.stack(delta_field, axis=0)
