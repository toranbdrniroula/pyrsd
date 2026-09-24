"""
pyrsd/utils/io.py 
utility for loading images and files into pyrsd.

Filesystem interactions are isolated here and everything else are completely matrix operations
"""

import cv2
import numpy as np
import re
import json
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".tif",".tiff",".png",".jpg",".jpeg",".bmp"})

SAT_THRESHOLD:float = 0.2
# Minimum HSV value (brightness, 0-1). Saturation is scale-free, so near-black pixels
# (e.g. RGB = 0,1,0) have S = 1 and pass the saturation test, but their hue is pure
# quantisation noise. 0.05 = 13/255 for 8-bit images.
VAL_THRESHOLD:float = 0.05

def _sort_key(s: str)->list:
    """breaks filename into list of chunks of alphabets and numbers"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)",s)]

def sequence_number(path: str) -> int | None:
    """extracts the last integer from the filename"""   
    s = re.search(r"(\d+)(?=\D*$)",Path(path).stem)
    return int(s.group(1)) if s else None

def find_images(folder: str) -> list[str]:
    """finds images in given directory and sorts them"""   
    base = Path(folder)
    if not base.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder")
    paths = [str(f) for f in base.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=_sort_key)

def load_image(path: str) -> np.ndarray:
    """loads image as BGR or BGRA unchanged"""
    img = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"{path} image could not be loaded")
    return img

def image_to_hue_field(img: np.ndarray, sat_threshold: float = SAT_THRESHOLD, val_threshold: float = VAL_THRESHOLD) -> np.ndarray:
    """extracts hue (degrees) from HSV of images.
    NaN where saturation < sat_threshold, value (brightness) < val_threshold, or pixel is transparent.
    Set val_threshold=0 to reproduce the old behaviour."""
    if img.ndim < 3:
        raise ValueError("Found Grayscale image! Expected: color image")

    alpha_mask = (img[:, :, 3]==0) if img.shape[2] == 4 else None

    bgr = img[:, :, :3]

    if img.dtype == np.uint16:
        bgr_float = bgr.astype(np.float32)/65535.0

    else:
        bgr_float = bgr.astype(np.float32)/255.0

    hsv = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    invalid = (sat < sat_threshold) | (val < val_threshold)
    if alpha_mask is not None:
        invalid = invalid | alpha_mask

    hue[invalid] = np.nan
    return hue

def load_json(path: str) -> dict:
    """loads json file and returns its content"""
    f = Path(path)
    if not f.exists():
        raise FileNotFoundError(f"{path} json not found")
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid json in path {path}: {e}") from e

def save_json(data: dict, path: str) -> None:
    """writes dictionary as json file to disk"""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_npy(path: str) -> np.ndarray:
    """loads numpy file"""
    f = Path(path)
    if not f.exists():
        raise FileNotFoundError(f"{path} npy file not found")
    return np.load(f)
    
def save_npy(array: np.ndarray, path: str) -> None:
    """saves numpy array as npy file to disk"""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, array)
