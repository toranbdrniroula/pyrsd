"""
pyrsd/core/solvers/integration.py
performs 1D integration in the flow field.
"""

import warnings
import numpy as np
from scipy.integrate import cumulative_trapezoid

def _select_segment(valid_idx: np.ndarray, max_gap: int | None, ref_side: str, ref_index: int | None) -> tuple[int, int]:
      """returns (first, last) index of the contiguous run of valid pixels to integrate.
      Runs separated by more than max_gap NaN pixels are treated as separate segments."""
      if max_gap is None:
            return int(valid_idx[0]), int(valid_idx[-1])
      breaks = np.where(np.diff(valid_idx) > max_gap + 1)[0]
      starts = np.r_[0, breaks + 1]
      ends = np.r_[breaks, len(valid_idx) - 1]
      segs = [(int(valid_idx[s]), int(valid_idx[e])) for s, e in zip(starts, ends)]
      if ref_side == "start":
            return segs[0]
      if ref_side == "end":
            return segs[-1]
      for a, b in segs:
            if a <= ref_index <= b:
                  return a, b
      return min(segs, key=lambda ab: min(abs(ref_index - ab[0]), abs(ref_index - ab[1])))

def integrate_1d(gradient_field: np.ndarray, dr_mm: float, ref_value: float, axis: int = 0, ref_side: str = "start", ref_index: int|None = None, bidirectional: bool = False, max_gap: int|None = 5) -> np.ndarray:
      """Performs 1 D culumative integration of a gradient field
      Parameters:
      ------------
      gradient_field: float64 ndarray 
      dr_mm: pixel spacing in mm along the integration axis
      ref_value: anchor value at the reference pixel 
      axis: 0 = integrates along rows (along y axis)
            1 = integrates along columns (along x axis)
      ref_side: 'start' - selects the first valid pixel of each line.
                  'end' - selects the last valid pixel of each line.
                  'index' - anchor at the pixel specified on ref_index.
      ref_index: pixel index used when ref_side='index' 
      bidirectional: deprecated, has no effect (see note below)
      max_gap: NaN gaps of up to max_gap pixels are bridged by linear interpolation. Longer gaps split the
               line; only the segment selected by ref_side is integrated (start -> first, end -> last,
               index -> the one containing ref_index) and the rest stays NaN. None reproduces the old
               behaviour of integrating from the first to the last valid pixel across any gap.

      Returns: 
      Scalar_Field : float64 ndarray, of same shape as gradient_field

      Note: averaging the forward and backward cumulative integrals only shifts the result by a constant
      (total/2), which the anchor shift then removes, so bidirectional=True was identical to False.
      """
      if ref_side not in ("start", "end", "index"):
            raise ValueError(f"ref_side must be 'start', 'end' or 'index', got '{ref_side}'")
      if ref_side == "index" and ref_index is None:
            raise ValueError("ref_side='index' requires ref_index")
      if bidirectional:
            warnings.warn("bidirectional has no effect after anchoring and is deprecated", DeprecationWarning, stacklevel=2)

      gradient = np.moveaxis(gradient_field.astype(np.float64), axis, 0)
      result = np.full_like(gradient, np.nan)

      for col in range(gradient.shape[1]):
            line = gradient[:,col]
            valid_idx = np.where(~np.isnan(line))[0]
            if valid_idx.size == 0:
                  continue

            i_start, i_end = _select_segment(valid_idx, max_gap, ref_side, ref_index)
            segment = line[i_start:i_end+1].copy()

            nan_mask = np.isnan(segment)
            if nan_mask.any():
                  x = np.arange(len(segment))
                  segment[nan_mask] = np.interp(x[nan_mask],x[~nan_mask],segment[~nan_mask])
            
            integrated = cumulative_trapezoid(segment, dx=dr_mm, initial=0.0)
            seg_len = len(integrated)
            anchor = seg_len-1

            if ref_side == "start":
                  anchor = 0  
            elif ref_side == "index":
                  anchor = max(0, min(ref_index - i_start, anchor))

            result[i_start:i_end+1, col] = integrated + (ref_value - integrated[anchor])

      return np.moveaxis(result, 0, axis)
