"""
pyrsd/core/calibration.py
extracts data for hue displacement calibration curve from sequential filter calibration images
"""

import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline
from pyrsd.utils.io import (find_images, load_image, image_to_hue_field, load_json, sequence_number)

def mean_hue_in_roi(hue_field: np.ndarray, roi_size: int) -> float:
    h, w = hue_field.shape[:2]
    mid = roi_size//2
    cy, cx = h//2, w//2
    top, bottom = max(0,cy-mid), min(h,cy+mid)
    left, right = max(0,cx-mid), min(w,cx+mid) 
    roi = hue_field[top:bottom, left:right]
    valid = roi[~np.isnan(roi)]
    if valid.size == 0:
        raise ValueError("No valid pixels in ROI")
    angles = np.deg2rad(valid)
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float(np.rad2deg(mean_angle) % 360)

def build_calibration_data(image_folder: str, step_size_mm: float, roi_size: int) -> list[dict]:
    """loads images and returns hue and displacement value data"""
    files = find_images(image_folder)
    results = []
    for path in files:
        seq = sequence_number(path)
        if seq is None:
            continue
        img = load_image(path)
        hue = image_to_hue_field(img)
        mean_hue = mean_hue_in_roi(hue, roi_size)
        displacement = float((seq) * step_size_mm)
        results.append({"hue":mean_hue,"displacement_mm":displacement})
    return results

def find_saturation_plateau(hue: np.ndarray, tol: float = 0.05, min_points: int = 3) -> np.ndarray:
    """boolean mask of calibration points lying on a hue plateau.
    A plateau is >= min_points sorted points whose consecutive hue differences are <= tol degrees
    (typically a channel clipped at the camera black level, e.g. hue stuck at exactly 240)."""
    hue = np.asarray(hue, dtype=float)
    mask = np.zeros(hue.shape, dtype=bool)
    if hue.size < min_points:
        return mask
    order = np.argsort(hue)
    close = np.diff(hue[order]) <= tol
    i = 0
    while i < len(close):
        if close[i]:
            j = i
            while j < len(close) and close[j]:
                j += 1
            if (j - i + 1) >= min_points:
                mask[order[i:j+1]] = True
            i = j
        else:
            i += 1
    return mask

def estimate_noise(hue: np.ndarray, displacement: np.ndarray, window: int = 7) -> float:
    """robust estimate (mm) of the scatter of displacement about a smooth curve of hue.
    Fits a quadratic in sliding windows of sorted points; the median residual variance is
    corrected for the chi-square(window-3) median. Needs monotone, plateau-free data."""
    order = np.argsort(hue)
    h, d = np.asarray(hue, float)[order], np.asarray(displacement, float)[order]
    dof = window - 3
    if len(h) < window or dof < 1:
        raise ValueError(f"need at least {window} points to estimate noise")
    var = []
    for i in range(len(h) - window + 1):
        x = h[i:i+window] - h[i:i+window].mean()
        y = d[i:i+window]
        coef = np.polyfit(x, y, 2)
        r = y - np.polyval(coef, x)
        var.append(float(r @ r) / dof)
    # median of chi2(dof)/dof
    from scipy.stats import chi2
    med = chi2.median(dof) / dof
    return float(np.sqrt(np.median(var) / med))

def fit_spline(hue: np.ndarray, displacement: np.ndarray, fit: str = "spline", hue_min: float = 0.0, hue_max: float = 360.0, sigma_mm: float | None = None, plateau: str = "mean"):
    """performs curve fitting using univariate spline

    fit:       'linear' | 'cubic' (both interpolate) | 'spline' (smoothing spline)
    sigma_mm:  displacement scatter (mm) used to set the smoothing factor s = N*sigma^2 for fit='spline'.
               None -> estimated from the data with estimate_noise(). The old behaviour (scipy default
               s = N, i.e. sigma = 1 mm) over-smooths a mm-scale calibration into a single cubic.
    plateau:   how to treat points on a saturated hue plateau (identical hue, different displacement):
               'mean' -> collapse to one point at the mean displacement (default; the half-width of the
                         plateau is kept as spline.plateau_halfwidth_mm = ambiguity of pixels at that hue)
               'drop' -> remove the points
               'keep' -> leave them (only valid for fit='spline'; interpolating fits will raise)
    The returned spline carries: .sigma_mm, .plateau_hue, .plateau_halfwidth_mm (None if no plateau).
    """
    if len(hue) < 2:
        raise RuntimeError(f"At least 2 calibration are required")
    if plateau not in ("mean", "drop", "keep"):
        raise ValueError("plateau must be 'mean', 'drop' or 'keep'")
    kwargs_fit = ("linear", "cubic", "spline")
    if fit not in kwargs_fit:
        raise ValueError(f"fit must be 'linear', 'cubic', or 'spline'. {fit} is not recognized")

    hue = np.asarray(hue, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    valid = (hue >= hue_min) & (hue <= hue_max)
    hue, displacement = hue[valid], displacement[valid]

    if len(hue) < 2:
        raise ValueError(f"Fewer than 2 points remain after trimming to hue range [{hue_min}, {hue_max}]. Check your hue_min and hue_max values.")

    plateau_hue, halfwidth = None, None
    pmask = find_saturation_plateau(hue)
    if pmask.any():
        plateau_hue = float(hue[pmask].mean())
        halfwidth = float(0.5 * (displacement[pmask].max() - displacement[pmask].min()))
        if plateau != "keep":
            warnings.warn(f"{int(pmask.sum())} calibration points lie on a hue plateau at {plateau_hue:.2f} deg "
                          f"(displacement {displacement[pmask].min():.3f}-{displacement[pmask].max():.3f} mm); "
                          f"pixels at this hue are ambiguous by +/-{halfwidth:.3f} mm. plateau='{plateau}'.", RuntimeWarning)
            keep = ~pmask
            if plateau == "mean":
                hue = np.r_[hue[keep], plateau_hue]
                displacement = np.r_[displacement[keep], displacement[pmask].mean()]
            else:
                hue, displacement = hue[keep], displacement[keep]

    order = np.argsort(hue)
    hue, displacement = hue[order], displacement[order]
    if len(hue) < 2:
        raise ValueError("Fewer than 2 points remain after plateau handling")

    sigma_used = None
    if fit == "spline":
        sigma_used = float(sigma_mm) if sigma_mm is not None else estimate_noise(hue, displacement)
        spl = UnivariateSpline(hue, displacement, k=3, s=len(hue) * sigma_used**2)
    else:
        if np.any(np.diff(hue) <= 0):
            raise ValueError("Interpolating fits need strictly increasing hue; duplicate hue values remain. "
                             "Use plateau='mean' or 'drop', or fit='spline'.")
        spl = UnivariateSpline(hue, displacement, k=1 if fit == "linear" else 3, s=0)

    spl.sigma_mm = sigma_used
    spl.plateau_hue = plateau_hue
    spl.plateau_halfwidth_mm = halfwidth
    return spl

def build_calibration_json(data: list[dict], image_folder: str, step_size_mm: float, roi_size: int, hue_min: float, hue_max: float, fit: str) -> dict:
    """builds content of calibration json file"""
    return {
        "header": {
            "image_folder": image_folder,
            "step_size_mm": step_size_mm,
            "roi_size_px":  roi_size,
            "n_points":     len(data),
            "fit":          fit,
        },
        "valid_hue_range": [hue_min, hue_max],
        "data": data,
    }

def load_spline_from_json(calib_path: str, fit: str = "spline", hue_min: float = 0.0, hue_max: float = 360.0, sigma_mm: float | None = None, plateau: str = "mean"):
    """takes calibration file and loads calibration curve"""
    filter_data = load_json(calib_path)
    
    hue = np.array([d["hue"] for d in filter_data["data"]])
    displacement = np.array([d["displacement_mm"] for d in filter_data["data"]])

    min_hue, max_hue = map(float,filter_data["valid_hue_range"])

    hue_min = max(min_hue,hue_min)
    hue_max = min(max_hue,hue_max)

    return fit_spline(hue, displacement, fit, hue_min, hue_max, sigma_mm=sigma_mm, plateau=plateau)
