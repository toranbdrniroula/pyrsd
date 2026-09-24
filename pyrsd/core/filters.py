"""
pyrsd/core/filters.py
optional image filtering and denoising for displacement fields.
apply before passing to solvers if your images are noisy.

NaN pixels are never used as data: gaussian_filter uses normalised convolution; the other filters
fill NaNs with the nearest valid value before filtering (the old zero-fill pulled values next to
masked regions, e.g. the wall shadow, towards 0). NaNs are restored afterwards.
"""

import numpy as np

def _fill_nearest(field: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    invalid = np.isnan(field)
    if not invalid.any():
        return field.copy()
    if invalid.all():
        return np.zeros_like(field)
    idx = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return field[tuple(idx)]

def gaussian_filter(field: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    invalid = np.isnan(field)
    weight = (~invalid).astype(np.float64)
    num = _gaussian_filter(np.where(invalid, 0.0, field), sigma)
    den = _gaussian_filter(weight, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[invalid] = np.nan
    return out

def median_filter(field: np.ndarray, size: int = 3) -> np.ndarray:
    from scipy.ndimage import median_filter as _median_filter
    invalid = np.isnan(field)
    out = _median_filter(_fill_nearest(field), size)
    out[invalid] = np.nan
    return out

def bilateral_filter(field: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    from cv2 import bilateralFilter
    invalid = np.isnan(field)
    filtered = bilateralFilter(_fill_nearest(field).astype(np.float32), d, sigma_color, sigma_space)
    result = filtered.astype(np.float64)
    result[invalid] = np.nan
    return result

def tv_denoise(field: np.ndarray, weight: float = 0.1) -> np.ndarray:
    from skimage.restoration import denoise_tv_chambolle
    invalid = np.isnan(field)
    out = denoise_tv_chambolle(_fill_nearest(field), weight)
    out[invalid] = np.nan
    return out
