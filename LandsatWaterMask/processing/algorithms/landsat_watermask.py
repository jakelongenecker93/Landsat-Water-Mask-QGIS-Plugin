# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
from datetime import datetime
import datetime as _dt
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from osgeo import gdal, ogr, osr

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterFolderDestination,
    QgsRasterLayer,
)

# Optional classes (newer QGIS only). We keep imports defensive so the
# plugin can run back to QGIS 3.10 (Qt5 / older Processing API).
try:
    from qgis.core import QgsProcessingParameterMatrix  # type: ignore
except Exception:
    QgsProcessingParameterMatrix = None  # type: ignore

try:
    from qgis.core import QgsProcessingOutputString  # type: ignore
except Exception:
    QgsProcessingOutputString = None  # type: ignore


gdal.UseExceptions()

_PATHROW_RE = re.compile(r"_(\d{3})(\d{3})_")

_ACQDATE_YMD_RE = re.compile(r"(\d{8})")  # YYYYMMDD anywhere
_ACQDATE_DOY_RE = re.compile(r"(\d{4})(\d{3})T")  # YYYYDOYThhmmss anywhere

def _parse_acq_date_from_string(s: str):
    """Extract acquisition date from a filename/layer name.
    Supports:
      - _YYYYMMDD_ (or .YYYYMMDD.)
      - .YYYYDOYThhmmss. (HLS-style), e.g., 2026001T163711
    Returns datetime.date or None.
    """
    if not s:
        return None
    # 1) YYYYMMDD tokens
    for m in _ACQDATE_YMD_RE.finditer(s):
        token = m.group(1)
        try:
            y = int(token[0:4]); mo = int(token[4:6]); d = int(token[6:8])
            return _dt.date(y, mo, d)
        except Exception:
            continue
    # 2) YYYYDOYThhmmss tokens
    m = _ACQDATE_DOY_RE.search(s)
    if m:
        try:
            y = int(m.group(1)); doy = int(m.group(2))
            return _dt.date(y, 1, 1) + _dt.timedelta(days=doy - 1)
        except Exception:
            return None
    return None


# --- Segmentation helpers ----------------------------------------------------

_MONTH_NAME_TO_NUM = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_NUM_TO_MON3 = {v: k.capitalize()[:3] for k, v in _MONTH_NAME_TO_NUM.items() if len(k) == 3}

def _parse_month_token(tok: str) -> int:
    tok = tok.strip()
    if not tok:
        raise ValueError("Empty month token")
    if tok.isdigit():
        m = int(tok)
        if 1 <= m <= 12:
            return m
        raise ValueError(f"Month number out of range: {tok}")
    key = re.sub(r"[^a-z]", "", tok.lower())
    if key in _MONTH_NAME_TO_NUM:
        return _MONTH_NAME_TO_NUM[key]
    raise ValueError(f"Unrecognized month token: {tok}")

def parse_month_segments(spec: str) -> list[dict]:
    """Parse month segments like:
    - 'Jan-Mar;Oct-Dec'
    - '1-3;10-12'
    - 'Nov-Feb' (wrap)
    Returns list of dicts: {start,end,wrap,months,label}.
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    parts = [p.strip() for p in re.split(r"[;,]", spec) if p.strip()]
    segs: list[dict] = []
    for p in parts:
        # normalize separators
        p_norm = p.replace("to", "-").replace("–", "-").replace("—", "-")
        if "-" in p_norm:
            a, b = [x.strip() for x in p_norm.split("-", 1)]
            s = _parse_month_token(a)
            e = _parse_month_token(b)
        else:
            s = e = _parse_month_token(p_norm)
        wrap = e < s
        if wrap:
            months = list(range(s, 13)) + list(range(1, e + 1))
        else:
            months = list(range(s, e + 1))
        label = f"{_NUM_TO_MON3.get(s, str(s))}-{_NUM_TO_MON3.get(e, str(e))}" if s != e else f"{_NUM_TO_MON3.get(s, str(s))}"
        segs.append({"start": s, "end": e, "wrap": wrap, "months": set(months), "label": label})
    return segs

def parse_year_segments(spec: str) -> list[dict]:
    """Parse year segments like: '1985-1989;2010-2020' or '2015;2020-2022'."""
    spec = (spec or "").strip()
    if not spec:
        return []
    parts = [p.strip() for p in re.split(r"[;,]", spec) if p.strip()]
    segs: list[dict] = []
    for p in parts:
        p_norm = p.replace("to", "-").replace("–", "-").replace("—", "-")
        if "-" in p_norm:
            a, b = [x.strip() for x in p_norm.split("-", 1)]
            ys, ye = int(a), int(b)
        else:
            ys = ye = int(p_norm)
        if ye < ys:
            ys, ye = ye, ys
        label = f"{ys}-{ye}" if ys != ye else f"{ys}"
        segs.append({"start": ys, "end": ye, "label": label})
    return segs

def _safe_suffix(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s)

# HLS Sentinel-2 style token: 2026001T163711 (YYYY + DOY + 'T' + HHMMSS)
_HLS_ACQ_RE = re.compile(r"\.(\d{7}T\d{6})\.")
# OPERA DSWx-HLS token: 20250327T213551Z (YYYYMMDD + "T" + HHMMSS + Z)
_OPERA_ACQ_RE = re.compile(r"_(\d{8})T\d{6}Z")



def _is_sentinel_fmask_name(name: str) -> bool:
    """Return True if filename looks like a Sentinel-2 HLS S30 Fmask."""
    n = (name or "").lower()
    # Typical: HLS.S30.T16RBU.2026001T163711.v2.0.Fmask.tif
    return ("hls.s30" in n or n.startswith("hls.s30")) and ("fmask" in n) and ("qa_pixel" not in n)



def _is_opera_dswx_name(name: str) -> bool:
    """Return True if filename looks like an OPERA L3 DSWx-HLS water mask (BWTR).

    Example:
      OPERA_L3_DSWx-HLS_T05VMG_20250524T214529Z_20250526T113803Z_S2B_30_v1.0_B02_BWTR.tif

    Notes (assumed/required by tool):
      - Byte (UInt8) raster
      - Water pixels == 1
      - NoData == 255
    """
    n = (name or '').lower()
    # Keep this intentionally permissive to accommodate minor naming variations
    return n.startswith('opera_') and ('dswx' in n) and ('bwtr' in n) and (n.endswith('.tif') or n.endswith('.tiff'))

class _ScaledFeedback:
    """Proxy feedback that maps 0..100 progress into a sub-range.

    This lets inner processing functions report smooth progress per group,
    while the outer algorithm still reports overall progress.
    """

    def __init__(self, fb, start: float, end: float, prefix: str = ""):
        self._fb = fb
        self._start = float(start)
        self._end = float(end)
        self._span = float(end) - float(start)
        self._prefix = prefix or ""

    def setProgress(self, percent: float):
        try:
            p = float(percent)
        except Exception:
            return
        if p < 0:
            p = 0.0
        elif p > 100:
            p = 100.0
        try:
            self._fb.setProgress(self._start + self._span * (p / 100.0))
        except Exception:
            pass

    def setProgressText(self, text: str):
        try:
            if hasattr(self._fb, "setProgressText"):
                self._fb.setProgressText(f"{self._prefix}{text}" if self._prefix else str(text))
        except Exception:
            pass

    def pushInfo(self, msg: str):
        try:
            self._fb.pushInfo(f"{self._prefix}{msg}" if self._prefix else str(msg))
        except Exception:
            pass

    def reportError(self, msg: str, fatalError: bool = False):
        try:
            self._fb.reportError(f"{self._prefix}{msg}" if self._prefix else str(msg), fatalError)
        except Exception:
            pass

    def isCanceled(self) -> bool:
        try:
            return bool(self._fb.isCanceled())
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._fb, name)

# -----------------------------
# Helpers
# -----------------------------

def _extract_acquisition_date(path_or_name) -> str:
    """Return acquisition date as YYYYMMDD when possible.

    Supports:
      - Landsat C2 naming (contains YYYYMMDD tokens)
      - HLS Sentinel-2 naming (contains YYYYDOYThhmmss tokens)
      - OPERA DSWx-HLS naming (contains YYYYMMDDThhmmssZ tokens)
    """
    name = Path(path_or_name).name if hasattr(path_or_name, "name") else str(path_or_name)

    # HLS first: convert YYYYDOY to YYYYMMDD
    m = _HLS_ACQ_RE.search(name)
    if m:
        token = m.group(1)  # e.g., 2026001T163711
        try:
            y = int(token[0:4])
            doy = int(token[4:7])
            d = datetime.strptime(f"{y}{doy:03d}", "%Y%j").strftime("%Y%m%d")
            return d
        except Exception:
            pass

    # OPERA: first YYYYMMDDThhmmssZ token -> YYYYMMDD
    if _is_opera_dswx_name(name):
        m_op = _OPERA_ACQ_RE.search(name)
        if m_op:
            return m_op.group(1)

    parts = name.split("_")

    # Find the path/row token like "023037" (6 digits), then take the NEXT 8-digit token as the date
    for i, p in enumerate(parts):
        if len(p) == 6 and p.isdigit():  # pathrow block
            if i + 1 < len(parts):
                d = parts[i + 1]
                if len(d) == 8 and d.isdigit():
                    return d

    # Fallback: first 8-digit token anywhere
    for p in parts:
        if len(p) == 8 and p.isdigit():
            return p

    return "UNKDATE"

def _group_paths_by_acqdate(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in paths:
        d = _extract_acquisition_date(p)
        groups.setdefault(d, []).append(p)
    for k in list(groups.keys()):
        groups[k] = _dedupe_paths(groups[k])
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))

def _dedupe_paths(paths: list[Path]) -> list[Path]:
    uniq = {}
    for p in paths:
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        uniq[key] = p
    return sorted(uniq.values())

def _extract_path_row(path_or_name: str | Path) -> tuple[str, str] | None:
    """Extract a spatial grouping key from common Landsat OR Sentinel-HLS filenames.

    Example:
      LC09_L2SP_022039_20251211_20251212_02_T1_...  -> ("022", "039")

    We look for:
      - Landsat: the first "_PPPRRR_" 6-digit token in the name.
      - HLS Sentinel-2: tile token like ".T16RBU." (returns ("S2", "T16RBU")).
    """
    name = Path(path_or_name).name if isinstance(path_or_name, Path) else str(path_or_name)

    # Sentinel-2 HLS tile
    if _is_sentinel_fmask_name(name):
        m2 = re.search(r"\.T(\d{2}[A-Z]{3})\.", name)
        if m2:
            return ("S2", f"T{m2.group(1)}")
        return ("S2", "UNK")

    # OPERA DSWx-HLS tile (e.g., _T05VMG_)
    if _is_opera_dswx_name(name):
        m3 = re.search(r"_T(\d{2}[A-Z]{3})_", name)
        if m3:
            return ('OPERA', f'T{m3.group(1)}')
        return ('OPERA', 'UNK')

    m = _PATHROW_RE.search(name)
    if not m:
        return None
    return m.group(1), m.group(2)

def _group_paths_by_pathrow(paths: list[Path]) -> dict[tuple[str, str], list[Path]]:
    groups: dict[tuple[str, str], list[Path]] = {}
    for p in paths:
        pr = _extract_path_row(p)
        if pr is None:
            # Keep "unknown" inputs together.
            pr = ("UNK", "UNK")
        groups.setdefault(pr, []).append(p)
    # stable ordering
    for k in list(groups.keys()):
        groups[k] = _dedupe_paths(groups[k])
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))

def _parse_rgb_triplet(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Background RGB must be 'r,g,b' (e.g., 0,0,0).")
    vals = tuple(int(p) for p in parts)
    for v in vals:
        if v < 0 or v > 255:
            raise ValueError("RGB values must be in 0..255.")
    return vals


def _parse_int_list(s: str, default: list[int]) -> list[int]:
    """Parse a comma-separated list of integers. Returns default if parsing yields no values."""
    if s is None:
        return list(default)
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    vals: list[int] = []
    for p in parts:
        try:
            vals.append(int(float(p)))
        except Exception:
            continue
    if not vals:
        return list(default)
    # De-duplicate while preserving order
    out: list[int] = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out

def _box_sum(mask: np.ndarray, k: int = 5) -> np.ndarray:
    # Fast kxk moving-window sum using an integral image.
    if k % 2 != 1:
        raise ValueError("k must be odd")
    pad = k // 2
    m = mask.astype(np.uint8)
    p = np.pad(m, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    # NOTE: We add an extra top row + left column of zeros to the integral image
    # so the kxk window sum returns the SAME shape as the original (un-padded) mask.
    s = p.cumsum(axis=0).cumsum(axis=1)
    s = np.pad(s, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    out = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    return out.astype(np.uint16)


def _pixel_majority_smooth(binary: np.ndarray, valid: np.ndarray | None, k: int) -> np.ndarray:
    """Fast pixel-based smoothing on a 0/1 mask using a majority filter.

    - k is the kernel size in pixels (odd). k=3 is a 3x3 majority filter.
    - valid (optional) restricts output to covered pixels (prevents expansion into NoData).

    Returns uint8 mask (0/1).
    """
    if binary is None:
        return binary

    try:
        k = int(k)
    except Exception:
        k = 3

    b = (binary > 0).astype(np.uint8)

    if valid is not None:
        v = (valid > 0)
        b = (b & v.astype(np.uint8))
    else:
        v = None

    if k <= 1:
        return b

    if k % 2 == 0:
        k += 1

    # Majority threshold
    thr = (k * k) // 2 + 1
    sm = (_box_sum(b, k=k) >= thr).astype(np.uint8)

    if v is not None:
        sm = (sm & v.astype(np.uint8))

    return sm


def _classify_rgb_land_water(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray,
    water_ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    """Classify RGB into water/land for REFL stacks.

    Returns a uint8 classification array:
      0 = excluded / not classified (background, nodata, etc.)
      1 = water
      2 = land

    `valid` must be a boolean mask indicating pixels eligible for classification.
    """
    # Clip before uint8 cast to avoid wraparound for odd NoData encodings.
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    valid = valid.astype(bool)

    all_lt_10 = (r < 10) & (g < 10) & (b < 10)

    zero_and_low = np.zeros(r.shape, dtype=bool)
    zero_and_low |= (r == 0) & ((g < 5) | (b < 5))
    zero_and_low |= (g == 0) & ((r < 5) | (b < 5))
    zero_and_low |= (b == 0) & ((r < 5) | (g < 5))

    near_zero_low = _box_sum(zero_and_low, k=5) > 0

    exclude = all_lt_10 | zero_and_low | near_zero_low

    # Anything outside `valid` must remain unclassified (0) so it doesn't become land by complement logic.
    non_class = (~valid) | exclude

    (rmin, rmax), (gmin, gmax), (bmin, bmax) = water_ranges
    water = (
        (r >= rmin) & (r <= rmax) &
        (g >= gmin) & (g <= gmax) &
        (b >= bmin) & (b <= bmax) &
        (~non_class)
    )

    cls = np.zeros(r.shape, dtype=np.uint8)
    cls[water] = 1
    cls[(~non_class) & (~water)] = 2
    return cls


def _srs_from_wkt(wkt: str) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    # IMPORTANT (GDAL>=3 / PROJ>=6): enforce traditional GIS axis order (lon,lat)
    # so EPSG:4326 transforms don't end up with swapped axes in some environments.
    try:
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    return srs

def _densified_bounds_in_target(ds: gdal.Dataset, target_srs: osr.SpatialReference, densify: int = 21) -> tuple[float, float, float, float]:
    # Approximate rasterio.transform_bounds(..., densify_pts=21) by sampling points along bounds.
    wkt = ds.GetProjection()
    if not wkt:
        raise ValueError(f"{ds.GetDescription()} has no projection/WKT.")
    src_srs = _srs_from_wkt(wkt)
    tx = osr.CoordinateTransformation(src_srs, target_srs)

    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    def pix2w(px, py):
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        return x, y

    corners = [pix2w(0,0), pix2w(cols,0), pix2w(cols,rows), pix2w(0,rows)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    pts = []
    for i in range(densify):
        t = i / (densify - 1) if densify > 1 else 0
        x = minx + t * (maxx - minx)
        pts.append((x, miny))
        pts.append((x, maxy))
        y = miny + t * (maxy - miny)
        pts.append((minx, y))
        pts.append((maxx, y))

    outx, outy = [], []
    for x, y in pts:
        xo, yo, _ = tx.TransformPoint(x, y)
        outx.append(xo)
        outy.append(yo)

    return min(outx), min(outy), max(outx), max(outy)

def _snap_bounds(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    xres: float,
    yres: float,
    anchor_x: float = 0.0,
    anchor_y: float = 0.0,
) -> tuple[float, float, float, float]:
    """Snap bounds to a pixel grid.

    GDAL geotransforms (and QGIS) treat the GT[0]/GT[3] origin as the *outer
    corner* of the top-left pixel.

    A common half-pixel shift happens when you snap to a grid anchored at 0,0
    while the source rasters are anchored at (origin_x + 0.5*res, origin_y -
    0.5*res) (i.e., mixing center-vs-corner conventions).

    To preserve exact alignment with source rasters, snap relative to an
    *anchor* (usually the reference raster's GT[0]/GT[3]).
    """

    xres = float(abs(xres))
    yres = float(abs(yres))

    left = anchor_x + math.floor((minx - anchor_x) / xres) * xres
    right = anchor_x + math.ceil((maxx - anchor_x) / xres) * xres
    bottom = anchor_y + math.floor((miny - anchor_y) / yres) * yres
    top = anchor_y + math.ceil((maxy - anchor_y) / yres) * yres
    return float(left), float(bottom), float(right), float(top)

def _warp_to_grid(
    ds: gdal.Dataset,
    dst_srs_wkt: str,
    bounds: tuple[float, float, float, float],
    xres: float,
    yres: float,
    width: int,
    height: int,
    resample: str = "near",
    dst_alpha: bool = True,
    out_dtype=None,
) -> gdal.Dataset:
    opts = gdal.WarpOptions(
        format="MEM",
        dstSRS=dst_srs_wkt,
        outputBounds=bounds,  # (minX, minY, maxX, maxY)
        xRes=xres,
        yRes=yres,
        width=width,
        height=height,
        resampleAlg=resample,
        dstAlpha=dst_alpha,
        multithread=True,
        outputType=out_dtype,
    )
    return gdal.Warp("", ds, options=opts)


def _warp_to_match(
    ds: gdal.Dataset,
    ref_gt,
    ref_wkt: str,
    width: int,
    height: int,
    resample_alg: str = "near",
) -> gdal.Dataset:
    """Warp *ds* to exactly match a reference grid (gt/wkt/size).

    This is a convenience wrapper used throughout the algorithm code.
    It preserves band NoData where possible so NoData does not become
    valid land/water in downstream counting/vectorization.
    """

    # Compute bounds from reference geotransform
    x0 = float(ref_gt[0])
    y0 = float(ref_gt[3])
    px_w = float(ref_gt[1])
    px_h = float(ref_gt[5])

    x1 = x0 + width * px_w
    y1 = y0 + height * px_h

    minx = min(x0, x1)
    maxx = max(x0, x1)
    miny = min(y0, y1)
    maxy = max(y0, y1)

    xres = abs(px_w)
    yres = abs(px_h)

    # Try to carry nodata through warping (works for QA_PIXEL and other single-band rasters)
    try:
        src_nodata = ds.GetRasterBand(1).GetNoDataValue()
    except Exception:
        src_nodata = None

    warp_kwargs = dict(
        format="MEM",
        dstSRS=ref_wkt,
        outputBounds=(minx, miny, maxx, maxy),
        xRes=xres,
        yRes=yres,
        width=int(width),
        height=int(height),
        resampleAlg=resample_alg,
        multithread=True,
        dstAlpha=True,
    )
    if src_nodata is not None:
        warp_kwargs["srcNodata"] = src_nodata
        warp_kwargs["dstNodata"] = src_nodata

    opts = gdal.WarpOptions(**warp_kwargs)
    return gdal.Warp("", ds, options=opts)



def _dataset_bounds(ds: gdal.Dataset) -> tuple[float, float, float, float]:
    """Return dataset bounds as (minx, miny, maxx, maxy) in the dataset CRS."""
    gt = ds.GetGeoTransform()
    w = ds.RasterXSize
    h = ds.RasterYSize

    # Corner coordinates
    def _pt(px: float, py: float) -> tuple[float, float]:
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        return float(x), float(y)

    corners = [_pt(0, 0), _pt(w, 0), _pt(0, h), _pt(w, h)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _transform_bounds(
    bounds: tuple[float, float, float, float],
    src_wkt: str,
    dst_wkt: str,
) -> tuple[float, float, float, float]:
    """Transform bounds (minx,miny,maxx,maxy) from src_wkt to dst_wkt."""
    if (src_wkt or '').strip() == (dst_wkt or '').strip():
        return bounds

    src = osr.SpatialReference()
    dst = osr.SpatialReference()
    src.ImportFromWkt(src_wkt)
    dst.ImportFromWkt(dst_wkt)

    # Traditional GIS axis order (x,y) for safety across GDAL/PROJ versions
    try:
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass

    ct = osr.CoordinateTransformation(src, dst)

    minx, miny, maxx, maxy = bounds
    pts = [
        (minx, miny),
        (minx, maxy),
        (maxx, miny),
        (maxx, maxy),
    ]
    xs, ys = [], []
    for x, y in pts:
        xx, yy, *_ = ct.TransformPoint(float(x), float(y))
        xs.append(xx)
        ys.append(yy)
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _compute_union_grid(
    paths: list[Path],
    ref_wkt: str,
    xres: float,
    yres: float,
    anchor_x: float | None = None,
    anchor_y: float | None = None,
    feedback=None,
    stack_label: str = "QA_PIXEL",
    label: str = "",
) -> tuple[tuple, int, int]:
    """Compute a reference (gt,width,height) covering the UNION extent of inputs.

    - Output CRS is ref_wkt.
    - Pixel sizes are (xres,yres) in ref_wkt units.
    - Bounds are aligned to the pixel grid.
    """

    if not paths:
        raise RuntimeError("No rasters provided.")

    minx = miny = maxx = maxy = None

    for p in paths:
        ds = gdal.Open(str(p))
        if ds is None:
            if feedback is not None:
                try:
                    feedback.pushInfo(f"{label}Skipping (open failed): {p}")
                except Exception:
                    pass
            continue

        b = _dataset_bounds(ds)
        src_wkt = ds.GetProjection()
        try:
            b = _transform_bounds(b, src_wkt, ref_wkt)
        except Exception:
            # If transform fails, fall back to assuming same CRS
            pass

        if minx is None:
            minx, miny, maxx, maxy = b
        else:
            minx = min(minx, b[0])
            miny = min(miny, b[1])
            maxx = max(maxx, b[2])
            maxy = max(maxy, b[3])

    if minx is None:
        raise RuntimeError("Could not compute bounds for inputs.")

    # Align to pixel grid (north-up assumed). IMPORTANT: snap relative to an
    # anchor (usually the first raster's GT[0]/GT[3]) to avoid half-pixel shifts.
    if anchor_x is None:
        anchor_x = 0.0
    if anchor_y is None:
        anchor_y = 0.0

    left, bottom, right, top = _snap_bounds(
        float(minx), float(miny), float(maxx), float(maxy), float(xres), float(yres),
        anchor_x=float(anchor_x),
        anchor_y=float(anchor_y),
    )

    xres = float(abs(xres))
    yres = float(abs(yres))
    width = int(max(1, math.ceil((right - left) / xres)))
    height = int(max(1, math.ceil((top - bottom) / yres)))

    gt = (float(left), xres, 0.0, float(top), 0.0, -yres)
    return gt, width, height

def _write_gtiff(path: Path, arr: np.ndarray, gt, proj_wkt: str, nodata=0):
    """Write a single-band GeoTIFF.

    - Uses Byte for 0..255 rasters.
    - Uses UInt16 automatically when values exceed 255 (e.g., per-pixel water counts).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    a = np.asarray(arr)
    try:
        maxv = int(np.nanmax(a))
    except Exception:
        maxv = 0

    if a.dtype in (np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64) or maxv > 255:
        gdal_type = gdal.GDT_UInt16
        out = np.clip(a, 0, 65535).astype(np.uint16)
    else:
        gdal_type = gdal.GDT_Byte
        out = np.clip(a, 0, 255).astype(np.uint8)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(path),
        int(out.shape[1]),
        int(out.shape[0]),
        1,
        gdal_type,
        options=["COMPRESS=LZW", "TILED=YES"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj_wkt)
    band = ds.GetRasterBand(1)
    band.WriteArray(out)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    ds.FlushCache()
    ds = None

def _resample_array_to_grid(
    src_arr: np.ndarray,
    src_gt,
    src_wkt: str,
    dst_gt,
    dst_wkt: str,
    dst_shape: tuple[int, int],
    nodata: int = 0,
) -> np.ndarray:
    """Resample a single-band array onto a target grid using GDAL MEM + nearest-neighbor.

    This is used to align BWTR to FMASK (or vice-versa) when users run PIXEL-mode SUM
    without Landsat inputs.
    """
    src_arr = np.asarray(src_arr)
    dst_h, dst_w = int(dst_shape[0]), int(dst_shape[1])

    # Pick a safe GDAL type for the array.
    try:
        maxv = int(np.nanmax(src_arr))
    except Exception:
        maxv = 0
    gdt = gdal.GDT_UInt16 if (src_arr.dtype != np.uint8 or maxv > 255) else gdal.GDT_Byte

    mem = gdal.GetDriverByName("MEM")

    src_ds = mem.Create("", int(src_arr.shape[1]), int(src_arr.shape[0]), 1, gdt)
    src_ds.SetGeoTransform(src_gt)
    src_ds.SetProjection(src_wkt)
    sb = src_ds.GetRasterBand(1)
    sb.WriteArray(src_arr)
    sb.SetNoDataValue(nodata)
    sb.FlushCache()

    dst_ds = mem.Create("", dst_w, dst_h, 1, gdt)
    dst_ds.SetGeoTransform(dst_gt)
    dst_ds.SetProjection(dst_wkt)
    db = dst_ds.GetRasterBand(1)
    db.Fill(nodata)
    db.SetNoDataValue(nodata)
    db.FlushCache()

    gdal.ReprojectImage(src_ds, dst_ds, src_wkt, dst_wkt, gdal.GRA_NearestNeighbour)

    out = dst_ds.GetRasterBand(1).ReadAsArray()
    src_ds = None
    dst_ds = None
    return out

def _delete_shp_family(shp_path: Path):
    base = shp_path.with_suffix("")
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".sbn", ".sbx", ".fix"]:
        p = base.with_suffix(ext)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def _chaikin_smooth_closed_ring_xy(xy: list[tuple[float, float]], iterations: int = 1, weight: float = 0.25) -> list[tuple[float, float]]:
    """Chaikin corner-cutting for a closed ring.

    - xy must include the closing vertex (i.e., first == last) OR will be treated as open and closed.
    - weight controls the cut distance; 0.25 is the classic Chaikin value.
    """
    if not xy or len(xy) < 4:
        return xy

    pts = list(xy)
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    w = float(weight)
    w = max(0.0, min(0.5, w))

    for _ in range(max(0, int(iterations))):
        new_pts: list[tuple[float, float]] = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            # Q = (1-w)P0 + wP1 ; R = wP0 + (1-w)P1
            qx = (1.0 - w) * x0 + w * x1
            qy = (1.0 - w) * y0 + w * y1
            rx = w * x0 + (1.0 - w) * x1
            ry = w * y0 + (1.0 - w) * y1
            new_pts.append((qx, qy))
            new_pts.append((rx, ry))
        # close
        if new_pts and new_pts[0] != new_pts[-1]:
            new_pts.append(new_pts[0])
        pts = new_pts

        # Guard: if we somehow end up with too few points, bail.
        if len(pts) < 4:
            break

    return pts


def _smoothify_ogr_geometry(geom: ogr.Geometry, iterations: int = 1, weight: float = 0.25) -> ogr.Geometry:
    """Smooth polygons/multipolygons using Chaikin corner cutting ("smoothify")."""
    if geom is None:
        return geom

    gtype = geom.GetGeometryType()

    def _smooth_polygon(poly: ogr.Geometry) -> ogr.Geometry:
        out = ogr.Geometry(ogr.wkbPolygon)
        # exterior ring
        ext = poly.GetGeometryRef(0)
        if ext is not None:
            xy = [(ext.GetX(i), ext.GetY(i)) for i in range(ext.GetPointCount())]
            sm = _chaikin_smooth_closed_ring_xy(xy, iterations=iterations, weight=weight)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in sm:
                ring.AddPoint_2D(float(x), float(y))
            out.AddGeometry(ring)

        # interior rings (holes)
        for r_i in range(1, poly.GetGeometryCount()):
            hole = poly.GetGeometryRef(r_i)
            if hole is None:
                continue
            xyh = [(hole.GetX(i), hole.GetY(i)) for i in range(hole.GetPointCount())]
            smh = _chaikin_smooth_closed_ring_xy(xyh, iterations=iterations, weight=weight)
            ring_h = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in smh:
                ring_h.AddPoint_2D(float(x), float(y))
            out.AddGeometry(ring_h)
        return out

    if gtype in (ogr.wkbPolygon, ogr.wkbPolygon25D):
        return _smooth_polygon(geom)

    if gtype in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
        out_mp = ogr.Geometry(ogr.wkbMultiPolygon)
        for i in range(geom.GetGeometryCount()):
            g = geom.GetGeometryRef(i)
            if g is None:
                continue
            out_mp.AddGeometry(_smooth_polygon(g))
        return out_mp

    # Fallback: only smooth polygonal geometry.
    return geom

def _polygonize_water_to_single_feature(
    binary_tif: Path,
    out_vec: Path,
    label: str = "Water",
    class_id: int = 1,
    smoothify: bool = False,
    smoothify_iters: int = 1,
    smoothify_weight: float = 0.25,
) -> bool:
    # Polygonize where raster==1, dissolve to one feature, write EPSG:4326 vector output (SHP/GPKG/GeoJSON).
    binary_tif = Path(binary_tif)
    out_vec = Path(out_vec)

    # Remove existing output
    if out_vec.suffix.lower() == ".shp":
        _delete_shp_family(out_vec)
    else:
        if out_vec.exists():
            try:
                out_vec.unlink()
            except Exception:
                pass

    src = gdal.Open(str(binary_tif))
    if src is None:
        raise RuntimeError(f"Could not open raster: {binary_tif}")

    proj_wkt = src.GetProjection()
    if not proj_wkt:
        raise RuntimeError("Raster has no projection; cannot write EPSG:4326 shapefile.")
    src_srs = _srs_from_wkt(proj_wkt)

    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    try:
        srs4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    coord_tx = osr.CoordinateTransformation(src_srs, srs4326)

    band = src.GetRasterBand(1)

    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("mem")
    mem_lyr = mem_ds.CreateLayer("poly", src_srs, geom_type=ogr.wkbPolygon)
    mem_lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))

    gdal.Polygonize(band, None, mem_lyr, 0, options=["8CONNECTED=8"])

    polys = ogr.Geometry(ogr.wkbMultiPolygon)
    mem_lyr.ResetReading()
    for feat in mem_lyr:
        if feat.GetFieldAsInteger("DN") != 1:
            continue
        g = feat.GetGeometryRef()
        if g is None:
            continue
        polys.AddGeometry(g.Clone())

    if polys.GetGeometryCount() == 0:
        src = None
        mem_ds = None
        return False

    dissolved = polys.UnionCascaded()
    if smoothify:
        try:
            dissolved = _smoothify_ogr_geometry(dissolved, iterations=smoothify_iters, weight=smoothify_weight)
            # try to clean up any minor self-intersections introduced by smoothing
            dissolved = dissolved.Buffer(0)
        except Exception:
            # If smoothing fails for any reason, fall back to the dissolved geometry.
            pass
    dissolved_4326 = dissolved.Clone()
    dissolved_4326.Transform(coord_tx)

    ext = out_vec.suffix.lower()
    if ext == ".shp":
        drv_name = "ESRI Shapefile"
        layer_name = out_vec.stem
    elif ext in (".geojson", ".json"):
        drv_name = "GeoJSON"
        layer_name = out_vec.stem
    else:
        drv_name = "GPKG"
        layer_name = out_vec.stem or "Water"

    out_driver = ogr.GetDriverByName(drv_name)
    if out_driver is None:
        raise RuntimeError(f"OGR driver not available: {drv_name}")

    out_ds = out_driver.CreateDataSource(str(out_vec))
    out_lyr = out_ds.CreateLayer(layer_name, srs4326, geom_type=ogr.wkbMultiPolygon)

    out_lyr.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))
    out_lyr.CreateField(ogr.FieldDefn("label", ogr.OFTString))

    out_feat = ogr.Feature(out_lyr.GetLayerDefn())
    out_feat.SetField("class_id", int(class_id))
    out_feat.SetField("label", str(label))
    out_feat.SetGeometry(dissolved_4326)
    out_lyr.CreateFeature(out_feat)

    out_feat = None
    out_ds = None
    mem_ds = None
    src = None
    
def _polygonize_binary_array_to_single_feature(
    binary: np.ndarray,
    gt,
    proj_wkt: str,
    out_vec: Path,
    label: str = "Water",
    class_id: int = 1,
    smoothify: bool = False,
    smoothify_iters: int = 1,
    smoothify_weight: float = 0.25,
    nodata: int = 0,
) -> bool:
    """Polygonize where binary==1, dissolve to one feature, write EPSG:4326 vector output."""
    out_vec = Path(out_vec)

    # Remove existing output
    if out_vec.suffix.lower() == ".shp":
        _delete_shp_family(out_vec)
    else:
        if out_vec.exists():
            try:
                out_vec.unlink()
            except Exception:
                pass

    if binary is None or binary.size == 0:
        return False

    h, w = int(binary.shape[0]), int(binary.shape[1])

    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(gt)
    mem.SetProjection(proj_wkt)
    band = mem.GetRasterBand(1)
    band.WriteArray(binary.astype(np.uint8))
    band.SetNoDataValue(nodata)
    band.FlushCache()

    if not proj_wkt:
        raise RuntimeError("Raster has no projection; cannot write EPSG:4326 shapefile.")
    src_srs = _srs_from_wkt(proj_wkt)

    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    try:
        srs4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    coord_tx = osr.CoordinateTransformation(src_srs, srs4326)

    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("mem_poly")
    mem_lyr = mem_ds.CreateLayer("poly", src_srs, geom_type=ogr.wkbPolygon)
    mem_lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))

    gdal.Polygonize(band, None, mem_lyr, 0, options=["8CONNECTED=8"])

    polys = ogr.Geometry(ogr.wkbMultiPolygon)
    mem_lyr.ResetReading()
    for feat in mem_lyr:
        if feat.GetFieldAsInteger("DN") != 1:
            continue
        g = feat.GetGeometryRef()
        if g is None:
            continue
        polys.AddGeometry(g.Clone())

    if polys.GetGeometryCount() == 0:
        mem_ds = None
        mem = None
        return False

    dissolved = polys.UnionCascaded()
    if smoothify:
        try:
            dissolved = _smoothify_ogr_geometry(dissolved, iterations=smoothify_iters, weight=smoothify_weight)
            dissolved = dissolved.Buffer(0)
        except Exception:
            pass

    dissolved_4326 = dissolved.Clone()
    dissolved_4326.Transform(coord_tx)

    ext = out_vec.suffix.lower()
    if ext == ".shp":
        drv_name = "ESRI Shapefile"
        layer_name = out_vec.stem
    elif ext in (".geojson", ".json"):
        drv_name = "GeoJSON"
        layer_name = out_vec.stem
    else:
        drv_name = "GPKG"
        layer_name = out_vec.stem or label

    out_driver = ogr.GetDriverByName(drv_name)
    if out_driver is None:
        raise RuntimeError(f"OGR driver not available: {drv_name}")

    out_ds = out_driver.CreateDataSource(str(out_vec))
    out_lyr = out_ds.CreateLayer(layer_name, srs4326, geom_type=ogr.wkbMultiPolygon)

    out_lyr.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))
    out_lyr.CreateField(ogr.FieldDefn("label", ogr.OFTString))

    out_feat = ogr.Feature(out_lyr.GetLayerDefn())
    out_feat.SetField("class_id", int(class_id))
    out_feat.SetField("label", str(label))
    out_feat.SetGeometry(dissolved_4326)
    out_lyr.CreateFeature(out_feat)

    out_feat = None
    out_ds = None
    mem_ds = None
    mem = None
    return True


def _ensure_multipolygon(geom: ogr.Geometry) -> ogr.Geometry:
    """Ensure output geometry is a MultiPolygon (in-place clone as needed)."""
    if geom is None:
        return geom
    gtype = geom.GetGeometryType()
    if gtype in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
        return geom
    if gtype in (ogr.wkbPolygon, ogr.wkbPolygon25D):
        mp = ogr.Geometry(ogr.wkbMultiPolygon)
        mp.AddGeometry(geom)
        return mp
    # Try to coerce (e.g., GeometryCollection)
    mp = ogr.Geometry(ogr.wkbMultiPolygon)
    for i in range(geom.GetGeometryCount()):
        g = geom.GetGeometryRef(i)
        if g is None:
            continue
        if g.GetGeometryType() in (ogr.wkbPolygon, ogr.wkbPolygon25D):
            mp.AddGeometry(g)
        elif g.GetGeometryType() in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
            for j in range(g.GetGeometryCount()):
                pg = g.GetGeometryRef(j)
                if pg is not None:
                    mp.AddGeometry(pg)
    return mp




def _ogr_total_vertex_count(geom: ogr.Geometry) -> int:
    """Approximate total vertex count across polygon rings (used for perf logging)."""
    if geom is None:
        return 0
    try:
        gname = (geom.GetGeometryName() or "").upper()
    except Exception:
        gname = ""

    def _poly_cnt(poly: ogr.Geometry) -> int:
        if poly is None:
            return 0
        c = 0
        try:
            n = poly.GetGeometryCount()
        except Exception:
            return 0
        for i in range(n):
            ring = poly.GetGeometryRef(i)
            if ring is None:
                continue
            try:
                c += ring.GetPointCount()
            except Exception:
                pass
        return c

    if gname == "POLYGON":
        return _poly_cnt(geom)
    if gname == "MULTIPOLYGON":
        total = 0
        try:
            n = geom.GetGeometryCount()
        except Exception:
            return 0
        for i in range(n):
            pg = geom.GetGeometryRef(i)
            total += _poly_cnt(pg)
        return total

    # GeometryCollection fallback: count polygonal members
    total = 0
    try:
        n = geom.GetGeometryCount()
    except Exception:
        return 0
    for i in range(n):
        g = geom.GetGeometryRef(i)
        if g is None:
            continue
        try:
            nm = (g.GetGeometryName() or "").upper()
        except Exception:
            nm = ""
        if nm == "POLYGON":
            total += _poly_cnt(g)
        elif nm == "MULTIPOLYGON":
            total += _ogr_total_vertex_count(g)
    return total


def _meters_to_degrees_tol_at_lat(meters: float, lat_deg: float) -> float:
    """Convert meters to an approximate degree tolerance at latitude (for EPSG:4326 simplification)."""
    try:
        lat = float(lat_deg)
    except Exception:
        lat = 0.0
    coslat = abs(math.cos(math.radians(lat)))
    coslat = max(coslat, 0.2)  # clamp near poles
    return float(meters) / (111_320.0 * coslat)

def _approx_pixel_size_m_from_gt_wkt(gt, wkt: str) -> float:
    """Approximate pixel size in meters from a geotransform and CRS WKT."""
    try:
        x = abs(float(gt[1]))
        y = abs(float(gt[5]))
        srs = _srs_from_wkt(wkt) if wkt else None
        if srs is not None and not srs.IsGeographic():
            units = float(srs.GetLinearUnits() or 1.0)  # meters per CRS unit
            return max(x, y) * units
        # geographic degrees -> meters (rough; good enough)
        return max(x, y) * 111320.0
    except Exception:
        try:
            return max(abs(float(gt[1])), abs(float(gt[5])))
        except Exception:
            return 30.0


def _binary_array_to_dissolved_geom_4326(
    binary: np.ndarray,
    gt,
    proj_wkt: str,
    smoothify: bool = False,
    smoothify_iters: int = 1,
    smoothify_weight: float = 0.25,
    nodata: int = 0,
) -> ogr.Geometry | None:
    """Polygonize where binary==1, dissolve, and return geometry in EPSG:4326."""
    if binary is None or binary.size == 0:
        return None

    h, w = int(binary.shape[0]), int(binary.shape[1])
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    mem.SetGeoTransform(gt)
    mem.SetProjection(proj_wkt)
    band = mem.GetRasterBand(1)
    band.WriteArray(binary.astype(np.uint8))
    band.SetNoDataValue(nodata)
    band.FlushCache()

    if not proj_wkt:
        return None

    src_srs = _srs_from_wkt(proj_wkt)
    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    try:
        srs4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    coord_tx = osr.CoordinateTransformation(src_srs, srs4326)

    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("mem_poly")
    mem_lyr = mem_ds.CreateLayer("poly", src_srs, geom_type=ogr.wkbPolygon)
    mem_lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))
    gdal.Polygonize(band, None, mem_lyr, 0, options=["8CONNECTED=8"])

    polys = ogr.Geometry(ogr.wkbMultiPolygon)
    mem_lyr.ResetReading()
    for feat in mem_lyr:
        if feat.GetFieldAsInteger("DN") != 1:
            continue
        g = feat.GetGeometryRef()
        if g is None:
            continue
        if g.GetGeometryType() in (ogr.wkbPolygon, ogr.wkbPolygon25D):
            polys.AddGeometry(g.Clone())
        elif g.GetGeometryType() in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
            for j in range(g.GetGeometryCount()):
                pg = g.GetGeometryRef(j)
                if pg is not None:
                    polys.AddGeometry(pg.Clone())

    if polys.GetGeometryCount() == 0:
        mem_ds = None
        mem = None
        return None

    dissolved = polys.UnionCascaded()
    if smoothify:
        try:
            dissolved = _smoothify_ogr_geometry(dissolved, iterations=smoothify_iters, weight=smoothify_weight)
            dissolved = dissolved.Buffer(0)
        except Exception:
            pass

    dissolved_4326 = dissolved.Clone()
    dissolved_4326.Transform(coord_tx)
    dissolved_4326 = _ensure_multipolygon(dissolved_4326)

    mem_ds = None
    mem = None
    return dissolved_4326


def _union_multipolygons(geoms: list[ogr.Geometry]) -> ogr.Geometry | None:
    """
    Robustly union a list of polygon/multipolygon geometries.

    We occasionally hit OGR/GEOS TopologyException (invalid rings, self-intersections),
    especially after smoothing/simplification. To keep long batch/segmented runs from
    failing, we attempt to repair invalid geometries (MakeValid/Buffer(0)) and retry
    union operations.

    Returns a MULTIPOLYGON (or None if nothing unionable).
    """
    if not geoms:
        return None

    def _make_valid(g: ogr.Geometry) -> ogr.Geometry | None:
        if g is None:
            return None
        if g.IsEmpty():
            return None
        # Try native validity repair first (GDAL/OGR 3+)
        try:
            if hasattr(g, "MakeValid"):
                gv = g.MakeValid()
                if gv is not None and not gv.IsEmpty():
                    g = gv
        except Exception:
            pass
        # Fallback: buffer(0) can fix minor self-intersections
        try:
            if not g.IsValid():
                gb = g.Buffer(0)
                if gb is not None and not gb.IsEmpty():
                    g = gb
        except Exception:
            pass
        return g

    def _ensure_multi(g: ogr.Geometry) -> ogr.Geometry | None:
        if g is None:
            return None
        try:
            gt = g.GetGeometryType()
        except Exception:
            return None

        # Force polygonal
        try:
            gm = ogr.ForceToMultiPolygon(g)
            if gm is not None and not gm.IsEmpty():
                return gm
        except Exception:
            pass

        # Handle geometry collections by extracting polygonal parts
        try:
            if g.GetGeometryCount() > 0:
                mp = ogr.Geometry(ogr.wkbMultiPolygon)
                for i in range(g.GetGeometryCount()):
                    part = g.GetGeometryRef(i)
                    part = _ensure_multi(part)
                    if part is None:
                        continue
                    # part is MultiPolygon: append each poly
                    for j in range(part.GetGeometryCount()):
                        mp.AddGeometry(part.GetGeometryRef(j))
                if mp.GetGeometryCount() > 0:
                    return mp
        except Exception:
            pass

        return None

    # Pre-clean & normalize to multipolygons
    cleaned: list[ogr.Geometry] = []
    for g in geoms:
        g = _make_valid(g)
        g = _ensure_multi(g)
        if g is not None and not g.IsEmpty() and g.GetGeometryCount() > 0:
            cleaned.append(g)

    if not cleaned:
        return None

    # Try fast cascaded union first
    try:
        mp = ogr.Geometry(ogr.wkbMultiPolygon)
        for g in cleaned:
            # g is multipolygon
            for i in range(g.GetGeometryCount()):
                pg = g.GetGeometryRef(i)
                if pg is not None:
                    mp.AddGeometry(pg.Clone())
        if mp.GetGeometryCount() == 0:
            return None
        out = mp.UnionCascaded()
        out = _make_valid(out) if out is not None else None
        out = _ensure_multi(out) if out is not None else None
        return out
    except Exception:
        # Fall back to iterative union with repair attempts
        out: ogr.Geometry | None = None
        for g in cleaned:
            if out is None:
                out = g.Clone()
                continue
            try:
                out = out.Union(g)
            except Exception:
                # Repair both operands and retry once
                out_fix = _make_valid(out)
                g_fix = _make_valid(g)
                out_fix = _ensure_multi(out_fix) if out_fix is not None else None
                g_fix = _ensure_multi(g_fix) if g_fix is not None else None
                if out_fix is None:
                    out = g_fix.Clone() if g_fix is not None else out
                    continue
                if g_fix is None:
                    out = out_fix
                    continue
                try:
                    out = out_fix.Union(g_fix)
                except Exception:
                    # Last resort: accumulate without union (keeps run alive)
                    mp = ogr.Geometry(ogr.wkbMultiPolygon)
                    for i in range(out_fix.GetGeometryCount()):
                        mp.AddGeometry(out_fix.GetGeometryRef(i).Clone())
                    for i in range(g_fix.GetGeometryCount()):
                        mp.AddGeometry(g_fix.GetGeometryRef(i).Clone())
                    out = mp

        out = _make_valid(out) if out is not None else None
        out = _ensure_multi(out) if out is not None else None
        return out

def _write_single_feature_geom_4326(
    out_vec: Path,
    geom_4326: ogr.Geometry,
    label: str,
    class_id: int,
):
    """Write a single-feature polygon vector file in EPSG:4326."""
    out_vec = Path(out_vec)

    # Remove existing output
    if out_vec.suffix.lower() == ".shp":
        _delete_shp_family(out_vec)
    else:
        if out_vec.exists():
            try:
                out_vec.unlink()
            except Exception:
                pass

    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    try:
        srs4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass

    geom_4326 = _ensure_multipolygon(geom_4326)

    ext = out_vec.suffix.lower()
    if ext == ".shp":
        drv_name = "ESRI Shapefile"
        layer_name = out_vec.stem
    elif ext in (".geojson", ".json"):
        drv_name = "GeoJSON"
        layer_name = out_vec.stem
    else:
        drv_name = "GPKG"
        layer_name = out_vec.stem or label

    out_driver = ogr.GetDriverByName(drv_name)
    if out_driver is None:
        raise RuntimeError(f"OGR driver not available: {drv_name}")

    out_ds = out_driver.CreateDataSource(str(out_vec))
    out_lyr = out_ds.CreateLayer(layer_name, srs4326, geom_type=ogr.wkbMultiPolygon)

    out_lyr.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))
    out_lyr.CreateField(ogr.FieldDefn("label", ogr.OFTString))

    out_feat = ogr.Feature(out_lyr.GetLayerDefn())
    out_feat.SetField("class_id", int(class_id))
    out_feat.SetField("label", str(label))
    out_feat.SetGeometry(geom_4326)
    out_lyr.CreateFeature(out_feat)

    out_feat = None
    out_ds = None


def _write_empty_vector_4326(
    out_vec: Path,
    label: str,
):
    """Write an empty polygon vector file in EPSG:4326 (schema only, no features)."""
    out_vec = Path(out_vec)

    # Remove existing output
    if out_vec.suffix.lower() == ".shp":
        _delete_shp_family(out_vec)
    else:
        try:
            if out_vec.exists():
                out_vec.unlink()
        except Exception:
            pass

    out_vec.parent.mkdir(parents=True, exist_ok=True)

    ext = out_vec.suffix.lower()
    if ext == ".shp":
        drv_name = "ESRI Shapefile"
        layer_name = out_vec.stem
    elif ext in (".geojson", ".json"):
        drv_name = "GeoJSON"
        layer_name = out_vec.stem
    else:
        drv_name = "GPKG"
        layer_name = out_vec.stem or label

    out_driver = ogr.GetDriverByName(drv_name)
    if out_driver is None:
        raise RuntimeError(f"OGR driver not available: {drv_name}")

    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)

    out_ds = out_driver.CreateDataSource(str(out_vec))
    out_lyr = out_ds.CreateLayer(layer_name, srs4326, geom_type=ogr.wkbMultiPolygon)

    out_lyr.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))
    out_lyr.CreateField(ogr.FieldDefn("label", ogr.OFTString))

    out_ds = None



def _find_refl_tifs(folder: Path) -> list[Path]:
    folder = Path(folder)
    candidates = []
    for ext in ("tif", "tiff", "TIF", "TIFF"):
        candidates.extend(folder.glob(f"*_refl.{ext}"))
    return _dedupe_paths(candidates)

def _find_pixel_tifs(folder: Path) -> list[Path]:
    folder = Path(folder)
    candidates = []
    for ext in ("tif", "tiff", "TIF", "TIFF"):
        candidates.extend(folder.glob(f"*QA_PIXEL*.{ext}"))
        candidates.extend(folder.glob(f"*PIXEL*.{ext}"))
    return _dedupe_paths(candidates)

def _landsat_sensor_code_from_name(name: str) -> str | None:
    """Return Landsat sensor/platform code from the filename.

    Convention used here: the 3rd and 4th characters in the filename are one of
    04, 05, 07, 08, or 09 (e.g., LT04..., LT05..., LE07..., LC08..., LC09...).
    """
    if not name:
        return None
    base = Path(name).name
    if len(base) < 4:
        return None
    code = base[2:4]
    if code in {"04", "05", "07", "08", "09"}:
        return code
    return None

def _pixel_water_value_for_file(name: str) -> int:
    """Map QA_PIXEL 'water' integer to the correct value per Landsat generation.

    - Landsat 4/5/7: water == 5504
    - Landsat 8/9:   water == 21952
    """
    code = _landsat_sensor_code_from_name(name) or ""
    if code in {"04", "05", "07"}:
        return 5504
    # default to Landsat 8/9 behavior
    return 21952



def _pixel_water_values_for_file(name: str, values_457: list[int], values_89: list[int]) -> list[int]:
    """Return list of QA_PIXEL integer values considered water for this filename."""
    code = _landsat_sensor_code_from_name(name) or ""
    if code in {"04", "05", "07"}:
        return list(values_457)
    return list(values_89)

def _qa_water_value_from_filename(name: str) -> int:
    """Backward-compatible alias for the QA_PIXEL water-value lookup.

    The canonical implementation is _pixel_water_value_for_file().
    """
    return _pixel_water_value_for_file(name)


def _qa_water_values_from_filename(name: str, values_457: list[int], values_89: list[int]) -> list[int]:
    """Return water-code list using the filename's Landsat sensor code."""
    return _pixel_water_values_for_file(name, values_457=values_457, values_89=values_89)

def _qa_pixel_valid_mask(qa: np.ndarray, alpha: np.ndarray | None, nodata) -> np.ndarray:
    """Return boolean valid mask for Landsat Collection 2 QA_PIXEL.

    IMPORTANT:
      QA_PIXEL is bit-packed. Pixels outside the scene footprint are typically marked
      with the FILL bit (bit 0) set, which often appears as value==1. Those MUST NOT
      be treated as land.

    Logic:
      - Start from warp alpha (if present) to exclude pixels outside the warped extent.
      - Exclude any pixel where the FILL bit is set: (qa & 1) != 0.
      - Exclude explicit nodata value if the band reports one.
    """
    if alpha is not None:
        valid = (alpha > 0)
    else:
        valid = np.ones(qa.shape, dtype=bool)

    # Bit 0 is the FILL flag in QA_PIXEL (no data / outside footprint)
    try:
        fill = ((qa.astype(np.uint32) & 1) != 0)
    except Exception:
        fill = (qa == 1)

    valid2 = valid & (~fill)

    # Some QA_PIXEL rasters also use value==1 explicitly as NoData (fill-only).
    # Even though this is covered by the FILL bit test above, enforce it here so
    # that value==1 never contributes to water or land counts/vectors.
    try:
        valid2 &= (qa != 1)
    except Exception:
        pass

    if nodata is not None:
        try:
            valid2 &= (qa != nodata)
        except Exception:
            pass

    return valid2



def _process_refl_stack(
    tifs: list[Path],
    out_water_tif: Path | None,
    out_land_tif: Path | None,
    out_water_vec: Path | None,
    out_land_vec: Path | None,
    bg_rgb: tuple[int, int, int],
    water_ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    feedback=None,
    refl_resample: str = "near",
    smoothify: bool = False,
    smoothify_iters: int = 3,
    smoothify_weight: float = 0.2,
    write_water_tiffs: bool = False,
    write_land_tiffs: bool = False,
    write_water_vec: bool = True,
    write_land_vec: bool = False,
    # Kept for backwards/forwards compatibility with older/newer guided UI wrappers.
    # This function performs *single-source* processing; SUM is handled by the caller.
    do_sum: bool = False,
    stack_label: str | None = None,
    **_ignored,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple, str, Path | None, Path | None]:
    """Process a stack of *_refl.tif rasters.

    Returns:
        water_count: uint16 array (# of unique acquisition dates classified as water)
        land_count:  uint16 array (# of unique acquisition dates classified as land)
        valid_bin:   uint8 array (1 where pixel is valid at least once, else 0)
        gt:          geotransform
        ref_wkt:     CRS WKT
        written_water_tif: path if written else None
        written_land_tif:  path if written else None

    NOTE: Counts are per unique acquisition date. Within a date, overlapping scenes
    are merged with WATER priority. NoData/background pixels do not contribute to
    either water or land counts.
    """

    # Group by acquisition date (YYYYMMDD in filename)
    date_groups = _group_paths_by_acqdate(tifs)

    # Progress init
    try:
        feedback.setProgress(0)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText("[REFL] Preparing stack…")
    except Exception:
        pass
    # Determine union target grid (covers ALL inputs; matches shapefile union extent)
    ref_ds = gdal.Open(str(tifs[0]))
    if ref_ds is None:
        raise RuntimeError(f"Could not open: {tifs[0]}")

    ref_gt0 = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    xres = abs(float(ref_gt0[1]))
    yres = abs(float(ref_gt0[5]))

    gt, width, height = _compute_union_grid(
        tifs,
        proj,
        xres,
        yres,
        anchor_x=float(ref_gt0[0]),
        anchor_y=float(ref_gt0[3]),
        feedback=feedback,
        label="[REFL] ",
    )

    # Initialize counts and validity
    water_count = np.zeros((height, width), dtype=np.uint16)
    land_count = np.zeros((height, width), dtype=np.uint16)
    valid_union = np.zeros((height, width), dtype=bool)

    feedback.pushInfo(f"[REFL] Unique acquisition dates: {len(date_groups)}")

    # Process each acquisition date
    for i, (acq_date, paths) in enumerate(sorted(date_groups.items()), start=1):
        if feedback.isCanceled():
            raise RuntimeError("Canceled.")

        # Progress within this stack
        try:
            n_dates = max(1, len(date_groups))
            p0 = 5.0 + 85.0 * (float(i - 1) / float(n_dates))
            feedback.setProgress(p0)
            if hasattr(feedback, "setProgressText"):
                feedback.setProgressText(f"[REFL] Processing date {i}/{n_dates}: {acq_date}")
        except Exception:
            pass

        feedback.pushInfo(f"[REFL]  • Date {i}/{len(date_groups)}: {acq_date} ({len(paths)} file(s))")

        date_water = np.zeros((height, width), dtype=bool)
        date_valid = np.zeros((height, width), dtype=bool)

        for path in paths:
            ds = gdal.Open(str(path))
            if ds is None:
                feedback.pushInfo(f"[REFL]    - Skipping (open failed): {path}")
                continue

            # Warp to reference grid if needed
            if (ds.RasterXSize != width) or (ds.RasterYSize != height) or (ds.GetGeoTransform() != gt) or (ds.GetProjection() != proj):
                ds = _warp_to_match(ds, gt, proj, width, height, resample_alg=refl_resample)

            # Read RGB
            r = ds.GetRasterBand(1).ReadAsArray()
            g = ds.GetRasterBand(2).ReadAsArray()
            b = ds.GetRasterBand(3).ReadAsArray()

            # Exclude background color AND explicit band nodata values (if provided)
            valid = ~((r == bg_rgb[0]) & (g == bg_rgb[1]) & (b == bg_rgb[2]))

            # Some reflectance stacks may carry a numeric NoData (e.g. -9999). Exclude it so it
            # doesn't become "land" by complement logic.
            nd1 = ds.GetRasterBand(1).GetNoDataValue()
            nd2 = ds.GetRasterBand(2).GetNoDataValue()
            nd3 = ds.GetRasterBand(3).GetNoDataValue()
            if (nd1 is not None) or (nd2 is not None) or (nd3 is not None):
                nd_mask = np.zeros_like(valid, dtype=bool)
                if nd1 is not None:
                    nd_mask |= (r == nd1)
                if nd2 is not None:
                    nd_mask |= (g == nd2)
                if nd3 is not None:
                    nd_mask |= (b == nd3)
                valid &= ~nd_mask

            cls = _classify_rgb_land_water(r, g, b, valid, water_ranges)
            # valid2 excludes non-classified pixels (cls==0)
            valid2 = valid & (cls != 0)

            is_water = valid2 & (cls == 1)

            # Merge within-date with WATER priority
            date_water |= is_water
            date_valid |= valid2

            valid_union |= valid2

        # Update counts per date
        water_count += date_water.astype(np.uint16)
        land_count += (date_valid & (~date_water)).astype(np.uint16)

    # Update progress after finishing this acquisition date
    try:
        n_dates = max(1, len(date_groups))
        p1 = 5.0 + 85.0 * (float(i) / float(n_dates))
        feedback.setProgress(p1)
    except Exception:
        pass


    # Build valid bin for vectorization
    valid_bin = valid_union.astype(np.uint8)


    try:
        feedback.setProgress(92)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText("[REFL] Writing count rasters…")
    except Exception:
        pass

    # Write TIFF(s)
    written_water_tif = None
    written_land_tif = None

    # Mask pixels that were never valid as NoData in the output count rasters.
    nodata_value = np.uint16(65535)
    invalid = ~valid_union
    if write_water_tiffs and out_water_tif is not None:
        out = water_count.copy()
        out[invalid] = nodata_value
        _write_gtiff(out_water_tif, out, gt, proj, nodata=int(nodata_value))
        written_water_tif = out_water_tif
    if write_land_tiffs and out_land_tif is not None:
        out = land_count.copy()
        out[invalid] = nodata_value
        _write_gtiff(out_land_tif, out, gt, proj, nodata=int(nodata_value))
        written_land_tif = out_land_tif

    # Vectors are handled at a higher level (processAlgorithm). Return counts and grid.

    try:
        feedback.setProgress(100)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText("[REFL] Done.")
    except Exception:
        pass

    return water_count, land_count, valid_bin, gt, proj, written_water_tif, written_land_tif

def _process_pixel_stack(
    tifs: list[Path],
    out_water_tif: Path | None = None,
    out_land_tif: Path | None = None,
    out_water_vec: Path | None = None,
    out_land_vec: Path | None = None,
    feedback=None,
    # Optional log prefix used by callers (harmless if unused by this function)
    label_tag: str = "",
    pixel_water_vals_457: Sequence[int] = (5504,),
    pixel_water_vals_89: Sequence[int] = (21952,),
    sentinel_water_mode: str = "all",
    sentinel_custom_values: Sequence[int] = (32, 96, 160, 224),
    # Compatibility: some callers pass stack_label; treat it as an alias of label_tag.
    stack_label: str | None = None,
    smoothify: bool = False,
    smoothify_iters: int = 3,
    smoothify_weight: float = 0.2,
    write_water_tiffs: bool = False,
    write_land_tiffs: bool = False,
    write_water_vec: bool = True,
    write_land_vec: bool = False,
    # Compatibility: some versions pass do_sum here. The per-source pixel stack does not
    # compute SUM itself; SUM is handled by the algorithm core.
    do_sum: bool = False,
    **_compat_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple, str, Path | None, Path | None]:
    """Process a stack of Landsat QA_PIXEL, Sentinel-2 HLS S30 Fmask, and/or OPERA DSWx-HLS BWTR rasters.

    Water identification:
      - Landsat QA_PIXEL: equality against mission-specific QA values
      - Sentinel-2 HLS Fmask (8-bit): user-selected water category
      - OPERA DSWx-HLS BWTR (8-bit): water==1 (NoData==255)

    Returns:
        water_count: uint16 array (# of unique acquisition dates classified as water)
        land_count:  uint16 array (# of unique acquisition dates classified as land)
        valid_bin:   uint8 array (1 where pixel is valid at least once, else 0)
        gt:          geotransform
        ref_wkt:     CRS WKT
        written_water_tif: path if written else None
        written_land_tif:  path if written else None

    NOTE: Counts are per unique acquisition date. Within a date, overlapping scenes
    are merged with WATER priority. NoData pixels do not contribute to either water
    or land counts.
    """

    # Back-compat: some callers send stack_label instead of label_tag.
    if (not label_tag) and stack_label:
        label_tag = f"[{stack_label}] "

    date_groups = _group_paths_by_acqdate(tifs)

    # Progress init
    try:
        feedback.setProgress(0)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText(f"{label_tag} Preparing stack…")
    except Exception:
        pass
    ref_ds = gdal.Open(str(tifs[0]))
    if ref_ds is None:
        raise RuntimeError(f"Could not open: {tifs[0]}")

    ref_gt0 = ref_ds.GetGeoTransform()
    ref_wkt = ref_ds.GetProjection()
    xres = abs(float(ref_gt0[1]))
    yres = abs(float(ref_gt0[5]))

    gt, width, height = _compute_union_grid(
        tifs,
        ref_wkt,
        xres,
        yres,
        anchor_x=float(ref_gt0[0]),
        anchor_y=float(ref_gt0[3]),
        feedback=feedback,
        label=f"{label_tag} ",
    )

    water_count = np.zeros((height, width), dtype=np.uint16)
    land_count = np.zeros((height, width), dtype=np.uint16)
    valid_union = np.zeros((height, width), dtype=bool)

    feedback.pushInfo(f"{label_tag} Unique acquisition dates: {len(date_groups)}")

    for i, (acq_date, paths) in enumerate(sorted(date_groups.items()), start=1):
        if feedback.isCanceled():
            raise RuntimeError("Canceled.")

        # Progress within this stack
        try:
            n_dates = max(1, len(date_groups))
            p0 = 5.0 + 85.0 * (float(i - 1) / float(n_dates))
            feedback.setProgress(p0)
            if hasattr(feedback, "setProgressText"):
                feedback.setProgressText(f"{label_tag} Processing date {i}/{n_dates}: {acq_date}")
        except Exception:
            pass

        feedback.pushInfo(f"{label_tag}  • Date {i}/{len(date_groups)}: {acq_date} ({len(paths)} file(s))")

        date_water = np.zeros((height, width), dtype=bool)
        date_valid = np.zeros((height, width), dtype=bool)

        for path in paths:
            ds = gdal.Open(str(path))
            if ds is None:
                feedback.pushInfo(f"{label_tag}    - Skipping (open failed): {path}")
                continue

            # Warp to reference grid if needed (QA is categorical: nearest)
            if (ds.RasterXSize != width) or (ds.RasterYSize != height) or (ds.GetGeoTransform() != gt) or (ds.GetProjection() != ref_wkt):
                ds = _warp_to_match(ds, gt, ref_wkt, width, height, resample_alg="near")

            qa = ds.GetRasterBand(1).ReadAsArray()
            nodata = ds.GetRasterBand(1).GetNoDataValue()

            alpha = None
            try:
                if hasattr(ds, "RasterCount") and ds.RasterCount and int(ds.RasterCount) > 1:
                    alpha = ds.GetRasterBand(int(ds.RasterCount)).ReadAsArray()
            except Exception:
                alpha = None

            alpha = None
            try:
                if hasattr(ds, "RasterCount") and ds.RasterCount and int(ds.RasterCount) > 1:
                    alpha = ds.GetRasterBand(int(ds.RasterCount)).ReadAsArray()
            except Exception:
                alpha = None

            is_sentinel = _is_sentinel_fmask_name(path.name)
            is_opera = _is_opera_dswx_name(path.name)

            # Mark that we successfully read at least one file of this augment type.
            # This controls whether we later write FMASK/BWTR count rasters and vectors.
            if is_sentinel:
                seen_fmask = True
            elif is_opera:
                seen_bwtr = True

            if is_sentinel:
                # Sentinel HLS Fmask is 8-bit categorical. User-specified NoData is 255.
                # NOTE: 255 is ALSO inside the "All Water" 224-255 bin in some docs,
                # but for this tool we always treat 255 as NoData (invalid).
                qa8 = qa.astype(np.uint8, copy=False)
                valid2 = (qa8 != np.uint8(255))

                if sentinel_water_mode == "pure":
                    water_mask = np.isin(qa8, np.array([32, 96, 160, 224], dtype=np.uint8))
                elif sentinel_water_mode == "custom":
                    # Clip to 0..255 and remove 255 if present
                    vals = [int(v) for v in sentinel_custom_values if 0 <= int(v) <= 255 and int(v) != 255]
                    water_mask = np.isin(qa8, np.array(vals, dtype=np.uint8)) if vals else np.zeros_like(valid2)
                else:
                    # "All water" bins: 32–63, 96–127, 160–191, 224–254 (exclude 255)
                    water_mask = (
                        ((qa8 >= 32) & (qa8 <= 63))
                        | ((qa8 >= 96) & (qa8 <= 127))
                        | ((qa8 >= 160) & (qa8 <= 191))
                        | ((qa8 >= 224) & (qa8 <= 254))
                    )

                is_water = valid2 & water_mask

            elif is_opera:
                seen_bwtr = True
                # OPERA L3 DSWx-HLS BWTR is 8-bit. Water == 1. NoData == 255.
                qa8 = qa.astype(np.uint8, copy=False)
                valid2 = (qa8 != np.uint8(255))
                is_water = valid2 & (qa8 == np.uint8(1))

            else:
                # Landsat QA_PIXEL
                water_vals = _qa_water_values_from_filename(
                    path.name, values_457=pixel_water_vals_457, values_89=pixel_water_vals_89
                )
                # Valid pixels: exclude explicit nodata and QA_PIXEL FILL flag (bit 0)
                valid2 = _qa_pixel_valid_mask(qa, alpha=alpha, nodata=nodata)
                is_water = valid2 & np.isin(qa, water_vals)

            date_water |= is_water
            date_valid |= valid2
            valid_union |= valid2

        water_count += date_water.astype(np.uint16)
        land_count += (date_valid & (~date_water)).astype(np.uint16)

    # Update progress after finishing this acquisition date
    try:
        n_dates = max(1, len(date_groups))
        p1 = 5.0 + 85.0 * (float(i) / float(n_dates))
        feedback.setProgress(p1)
    except Exception:
        pass


    valid_bin = valid_union.astype(np.uint8)

    written_water_tif = None
    written_land_tif = None


    try:
        feedback.setProgress(92)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText(f"{label_tag} Writing count rasters…")
    except Exception:
        pass

    # Mask pixels that were never valid as NoData in the output count rasters.
    nodata_value = np.uint16(65535)
    invalid = ~valid_union
    if write_water_tiffs and out_water_tif is not None:
        out = water_count.copy()
        out[invalid] = nodata_value
        _write_gtiff(out_water_tif, out, gt, ref_wkt, nodata=int(nodata_value))
        written_water_tif = out_water_tif
    if write_land_tiffs and out_land_tif is not None:
        out = land_count.copy()
        out[invalid] = nodata_value
        _write_gtiff(out_land_tif, out, gt, ref_wkt, nodata=int(nodata_value))
        written_land_tif = out_land_tif


    try:
        feedback.setProgress(100)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText(f"{label_tag} Done.")
    except Exception:
        pass

    return water_count, land_count, valid_bin, gt, ref_wkt, written_water_tif, written_land_tif

def _process_both_by_acqdate(
    tifs_refl: list[Path],
    tifs_pixel: list[Path],
    tifs_augment: list[Path],
    out_refl_water_tif: Path | None,
    out_refl_land_tif: Path | None,
    out_pixel_water_tif: Path | None,
    out_pixel_land_tif: Path | None,
    out_fmask_water_tif: Path | None,
    out_fmask_land_tif: Path | None,
    out_bwtr_water_tif: Path | None,
    out_bwtr_land_tif: Path | None,
    out_sum_water_tif: Path | None,
    out_sum_land_tif: Path | None,
    bg_rgb: tuple[int, int, int],
    water_ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    pixel_water_vals_457: Sequence[int] = (5504,),
    pixel_water_vals_89: Sequence[int] = (21952,),
    sentinel_water_mode: str = "all",
    sentinel_custom_values: Sequence[int] = (32, 96, 160, 224),
    feedback=None,
    do_sum: bool = True,
    refl_resample: str = "near",
    write_water_tiffs: bool = False,
    write_land_tiffs: bool = False,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray | None, np.ndarray | None,
    tuple, str, dict,
]:
    """Process REFL + QA_PIXEL stacks together, grouped by acquisition date.

    Returns:
        refl_water_count, refl_land_count,
        pixel_water_count, pixel_land_count,
        fmask_water_count, fmask_land_count,
        bwtr_water_count, bwtr_land_count,
        sum_water_count (optional), sum_land_count (optional),
        gt, ref_wkt, written_paths

    Notes:
        - Counts are per unique acquisition date.
        - Within each date and within each pipeline, overlapping scenes are merged first (boolean OR),
          so a pixel/date contributes at most 1 count per pipeline.
        - SUM outputs are computed per unique acquisition date (0..Ndates):
            * Water per date: 1 if REFL OR QA_PIXEL is water (water priority)
            * Land  per date: inverse of SUM water within pixels valid in REFL OR QA_PIXEL for that date
        - NoData/background pixels do not contribute to either water or land counts.
    """

    # Label for log messages / output naming within BOTH pipeline
    # (This function processes Landsat REFL + Landsat QA_PIXEL stacks.)
    label = "QA_PIXEL"
    label_tag = f"[{label}]"

    # Group by date for each pipeline
    refl_by_date = _group_paths_by_acqdate(tifs_refl)
    pix_by_date = _group_paths_by_acqdate(tifs_pixel)
    aug_by_date = _group_paths_by_acqdate(tifs_augment)
    all_dates = sorted(set(refl_by_date.keys()) | set(pix_by_date.keys()) | set(aug_by_date.keys()))

    # Progress init
    try:
        feedback.setProgress(0)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText("[BOTH] Preparing stacks…")
    except Exception:
        pass

    # Reference grid: prefer REFL, then QA_PIXEL/FMASS/BWTR (pixel-like / augment).
    # In segmented runs it's valid to have *only* augment rasters (e.g., BWTR-only), so we must not assume PIXEL exists.
    if tifs_refl:
        ref_path = tifs_refl[0]
    elif tifs_pixel:
        ref_path = tifs_pixel[0]
    elif tifs_augment:
        ref_path = tifs_augment[0]
    else:
        raise RuntimeError("No raster inputs found for this segment (REFL/QA_PIXEL/FMASS/BWTR).")
    ref_ds = gdal.Open(str(ref_path))
    if ref_ds is None:
        raise RuntimeError(f"Could not open: {ref_path}")
    # Union target grid across BOTH pipelines so output rasters cover the full processed extent
    ref_gt0 = ref_ds.GetGeoTransform()
    ref_wkt = ref_ds.GetProjection()
    xres = abs(float(ref_gt0[1]))
    yres = abs(float(ref_gt0[5]))

    all_paths = list(tifs_refl) + list(tifs_pixel) + list(tifs_augment)
    gt, width, height = _compute_union_grid(
        all_paths,
        ref_wkt,
        xres,
        yres,
        anchor_x=float(ref_gt0[0]),
        anchor_y=float(ref_gt0[3]),
        feedback=feedback,
        label="[BOTH] ",
    )

    refl_water_count = np.zeros((height, width), dtype=np.uint16)
    pixel_water_count = np.zeros((height, width), dtype=np.uint16)
    fmask_water_count = np.zeros((height, width), dtype=np.uint16)
    bwtr_water_count = np.zeros((height, width), dtype=np.uint16)
    sum_water_count = np.zeros((height, width), dtype=np.uint16) if do_sum else None

    refl_land_count = np.zeros((height, width), dtype=np.uint16)
    pixel_land_count = np.zeros((height, width), dtype=np.uint16)
    fmask_land_count = np.zeros((height, width), dtype=np.uint16)
    bwtr_land_count = np.zeros((height, width), dtype=np.uint16)
    sum_land_count = np.zeros((height, width), dtype=np.uint16) if do_sum else None

    
    seen_fmask = False
    seen_bwtr = False
    valid_union = np.zeros((height, width), dtype=bool)

    feedback.pushInfo(f"[BOTH] Unique acquisition dates: {len(all_dates)}")

    for i, acq_date in enumerate(all_dates, start=1):
        if feedback.isCanceled():
            raise RuntimeError("Canceled.")

        # Progress within this stack
        try:
            n_dates = max(1, len(all_dates))
            p0 = 5.0 + 85.0 * (float(i - 1) / float(n_dates))
            feedback.setProgress(p0)
            if hasattr(feedback, "setProgressText"):
                feedback.setProgressText(f"[BOTH] Processing date {i}/{n_dates}: {acq_date}")
        except Exception:
            pass

        refl_paths = refl_by_date.get(acq_date, [])
        pix_paths = pix_by_date.get(acq_date, [])
        aug_paths = aug_by_date.get(acq_date, [])
        # Split augment stack into FMASK vs BWTR so they can be counted/written separately
        fmask_paths = [p for p in aug_paths if _is_sentinel_fmask_name(p.name)]
        bwtr_paths = [p for p in aug_paths if _is_opera_dswx_name(p.name)]
        other_aug_paths = [p for p in aug_paths if (p not in fmask_paths) and (p not in bwtr_paths)]
        feedback.pushInfo(f"[BOTH]  • Date {i}/{len(all_dates)}: {acq_date} (REFL={len(refl_paths)} QA_PIXEL={len(pix_paths)} FMASK={len(fmask_paths)} BWTR={len(bwtr_paths)})")

        date_refl_water = np.zeros((height, width), dtype=bool)
        date_refl_valid = np.zeros((height, width), dtype=bool)
        date_pix_water = np.zeros((height, width), dtype=bool)
        date_pix_valid = np.zeros((height, width), dtype=bool)
        date_aug_water = np.zeros((height, width), dtype=bool)
        date_aug_valid = np.zeros((height, width), dtype=bool)

        # --- REFL ---
        for path in refl_paths:
            ds = gdal.Open(str(path))
            if ds is None:
                feedback.pushInfo(f"[REFL]    - Skipping (open failed): {path}")
                continue

            if (ds.RasterXSize != width) or (ds.RasterYSize != height) or (ds.GetGeoTransform() != gt) or (ds.GetProjection() != ref_wkt):
                ds = _warp_to_match(ds, gt, ref_wkt, width, height, resample_alg=refl_resample)

            r = ds.GetRasterBand(1).ReadAsArray()
            g = ds.GetRasterBand(2).ReadAsArray()
            b = ds.GetRasterBand(3).ReadAsArray()

            valid = ~((r == bg_rgb[0]) & (g == bg_rgb[1]) & (b == bg_rgb[2]))

            nd1 = ds.GetRasterBand(1).GetNoDataValue()
            nd2 = ds.GetRasterBand(2).GetNoDataValue()
            nd3 = ds.GetRasterBand(3).GetNoDataValue()
            if (nd1 is not None) or (nd2 is not None) or (nd3 is not None):
                nd_mask = np.zeros_like(valid, dtype=bool)
                if nd1 is not None:
                    nd_mask |= (r == nd1)
                if nd2 is not None:
                    nd_mask |= (g == nd2)
                if nd3 is not None:
                    nd_mask |= (b == nd3)
                valid &= ~nd_mask

            cls = _classify_rgb_land_water(r, g, b, valid, water_ranges)
            valid2 = valid & (cls != 0)
            is_water = valid2 & (cls == 1)

            date_refl_water |= is_water
            date_refl_valid |= valid2
            valid_union |= valid2

        # --- PIXEL (Landsat QA_PIXEL and/or Sentinel HLS Fmask) ---
        for path in pix_paths:
            ds = gdal.Open(str(path))
            if ds is None:
                feedback.pushInfo(f"{label_tag}    - Skipping (open failed): {path}")
                continue

            if (ds.RasterXSize != width) or (ds.RasterYSize != height) or (ds.GetGeoTransform() != gt) or (ds.GetProjection() != ref_wkt):
                ds = _warp_to_match(ds, gt, ref_wkt, width, height, resample_alg="near")

            qa = ds.GetRasterBand(1).ReadAsArray()
            nodata = ds.GetRasterBand(1).GetNoDataValue()

            alpha = None
            try:
                if hasattr(ds, "RasterCount") and ds.RasterCount and int(ds.RasterCount) > 1:
                    alpha = ds.GetRasterBand(int(ds.RasterCount)).ReadAsArray()
            except Exception:
                alpha = None

            is_sentinel = _is_sentinel_fmask_name(path.name)
            is_opera = _is_opera_dswx_name(path.name)

            # Mark that we successfully opened at least one file of each augment type.
            # These flags control whether FMASK/BWTR outputs are written at the end.
            if is_sentinel:
                seen_fmask = True
            if is_opera:
                seen_bwtr = True

            if is_sentinel:
                qa8 = qa.astype(np.uint8, copy=False)
                valid2 = (qa8 != np.uint8(255))

                if sentinel_water_mode == "pure":
                    water_mask = np.isin(qa8, np.array([32, 96, 160, 224], dtype=np.uint8))
                elif sentinel_water_mode == "custom":
                    vals = [int(v) for v in sentinel_custom_values if 0 <= int(v) <= 255 and int(v) != 255]
                    water_mask = np.isin(qa8, np.array(vals, dtype=np.uint8)) if vals else np.zeros_like(valid2)
                else:
                    water_mask = (
                        ((qa8 >= 32) & (qa8 <= 63))
                        | ((qa8 >= 96) & (qa8 <= 127))
                        | ((qa8 >= 160) & (qa8 <= 191))
                        | ((qa8 >= 224) & (qa8 <= 254))
                    )

                is_water = valid2 & water_mask

            elif is_opera:
                # OPERA L3 DSWx-HLS BWTR is 8-bit. Water == 1. NoData == 255.
                qa8 = qa.astype(np.uint8, copy=False)
                valid2 = (qa8 != np.uint8(255))
                is_water = valid2 & (qa8 == np.uint8(1))

            else:
                water_vals = _qa_water_values_from_filename(
                    path.name, values_457=list(pixel_water_vals_457), values_89=list(pixel_water_vals_89)
                )

                # Valid pixels: exclude explicit nodata and QA_PIXEL FILL flag (bit 0)
                valid2 = _qa_pixel_valid_mask(qa, alpha=alpha, nodata=nodata)
                is_water = valid2 & np.isin(qa, water_vals)

            date_pix_water |= is_water
            date_pix_valid |= valid2
            valid_union |= valid2


        # --- FMASK / BWTR (augment) ---
        # These stacks can optionally be included in SUM, and can also be written out independently.
        date_fmask_water = np.zeros((height, width), dtype=bool)
        date_fmask_valid = np.zeros((height, width), dtype=bool)
        date_bwtr_water = np.zeros((height, width), dtype=bool)
        date_bwtr_valid = np.zeros((height, width), dtype=bool)

        def _accum_aug(path: Path, kind: str):
            nonlocal date_aug_water, date_aug_valid, valid_union, date_fmask_water, date_fmask_valid, date_bwtr_water, date_bwtr_valid, seen_fmask, seen_bwtr

            ds = gdal.Open(str(path))
            if ds is None:
                feedback.pushInfo(f"[{kind}]    - Skipping (open failed): {path}")
                return

            if (ds.RasterXSize != width) or (ds.RasterYSize != height) or (ds.GetGeoTransform() != gt) or (ds.GetProjection() != ref_wkt):
                ds = _warp_to_match(ds, gt, ref_wkt, width, height, resample_alg="near")

            qa = ds.GetRasterBand(1).ReadAsArray()
            nodata = ds.GetRasterBand(1).GetNoDataValue()

            alpha = None
            try:
                if hasattr(ds, "RasterCount") and ds.RasterCount and int(ds.RasterCount) > 1:
                    alpha = ds.GetRasterBand(int(ds.RasterCount)).ReadAsArray()
            except Exception:
                alpha = None

            is_sentinel = _is_sentinel_fmask_name(path.name)
            is_opera = _is_opera_dswx_name(path.name)
            # Mark whether we actually saw/read any FMASK/BWTR inputs so downstream writers can emit outputs only when present.
            if is_sentinel:
                seen_fmask = True
            if is_opera:
                seen_bwtr = True


            if is_sentinel:
                qa8 = qa.astype(np.uint8, copy=False)
                valid2 = (qa8 != np.uint8(255))

                if sentinel_water_mode == "pure":
                    water_mask = np.isin(qa8, np.array([32, 96, 160, 224], dtype=np.uint8))
                elif sentinel_water_mode == "custom":
                    vals = [int(v) for v in sentinel_custom_values if 0 <= int(v) <= 255 and int(v) != 255]
                    water_mask = np.isin(qa8, np.array(vals, dtype=np.uint8)) if vals else np.zeros_like(valid2)
                else:
                    water_mask = (
                        ((qa8 >= 32) & (qa8 <= 63))
                        | ((qa8 >= 96) & (qa8 <= 127))
                        | ((qa8 >= 160) & (qa8 <= 191))
                        | ((qa8 >= 224) & (qa8 <= 254))
                    )

                is_water = valid2 & water_mask
            elif is_opera:
                qa8 = qa.astype(np.uint8, copy=False)
                valid2 = (qa8 != np.uint8(255))
                is_water = valid2 & (qa8 == np.uint8(1))
            else:
                # Fallback: treat as QA_PIXEL rules
                water_vals = _qa_water_values_from_filename(
                    path.name, values_457=list(pixel_water_vals_457), values_89=list(pixel_water_vals_89)
                )
                valid2 = _qa_pixel_valid_mask(qa, alpha=alpha, nodata=nodata)
                is_water = valid2 & np.isin(qa, water_vals)

            # For SUM (if enabled), keep a combined AUG contribution
            date_aug_water |= is_water
            date_aug_valid |= valid2
            valid_union |= valid2

            # Also keep per-source contributions
            if is_sentinel:
                date_fmask_water |= is_water
                date_fmask_valid |= valid2
            elif is_opera:
                date_bwtr_water |= is_water
                date_bwtr_valid |= valid2

        for path in fmask_paths:
            _accum_aug(path, "FMASK")
        for path in bwtr_paths:
            _accum_aug(path, "BWTR")
        for path in other_aug_paths:
            _accum_aug(path, "AUG")

        # Update counts per-date
        refl_water_count += date_refl_water.astype(np.uint16)
        refl_land_count += (date_refl_valid & (~date_refl_water)).astype(np.uint16)

        pixel_water_count += date_pix_water.astype(np.uint16)
        pixel_land_count += (date_pix_valid & (~date_pix_water)).astype(np.uint16)

        fmask_water_count += date_fmask_water.astype(np.uint16)
        fmask_land_count += (date_fmask_valid & (~date_fmask_water)).astype(np.uint16)

        bwtr_water_count += date_bwtr_water.astype(np.uint16)
        bwtr_land_count += (date_bwtr_valid & (~date_bwtr_water)).astype(np.uint16)

        # Update progress after finishing this acquisition date
        try:
            n_dates = max(1, len(all_dates))
            p1 = 5.0 + 85.0 * (float(i) / float(n_dates))
            feedback.setProgress(p1)
        except Exception:
            pass

        if do_sum and (sum_water_count is not None) and (sum_land_count is not None):
            # SUM outputs are computed per acquisition date (0..Ndates):
            #   - Water: counts 1 for a date if EITHER pipeline flags water that date.
            #   - Land : inverse of SUM water within pixels valid in EITHER pipeline that date.
            # Per-source land = valid AND not water; sum counts across sources for this acquisition date
            date_refl_land = date_refl_valid & (~date_refl_water)
            date_pix_land  = date_pix_valid  & (~date_pix_water)
            date_aug_land  = date_aug_valid  & (~date_aug_water)

            date_sum_water_count = (
                date_refl_water.astype(np.uint16)
                + date_pix_water.astype(np.uint16)
                + date_aug_water.astype(np.uint16)
            )
            date_sum_land_count = (
                date_refl_land.astype(np.uint16)
                + date_pix_land.astype(np.uint16)
                + date_aug_land.astype(np.uint16)
            )

            sum_water_count += date_sum_water_count
            sum_land_count += date_sum_land_count

    valid_bin = valid_union.astype(np.uint8)

    written = {
        "refl_water": None,
        "refl_land": None,
        "pixel_water": None,
        "pixel_land": None,
        "fmask_water": None,
        "fmask_land": None,
        "bwtr_water": None,
        "bwtr_land": None,
        "sum_water": None,
        "sum_land": None,
    }

    try:
        feedback.setProgress(92)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText("[BOTH] Writing count rasters…")
    except Exception:
        pass

    # Write requested rasters (mask never-valid pixels as NoData so they don't appear as land/water)
    nodata_value = np.uint16(65535)
    invalid_refl = ~((refl_water_count + refl_land_count) > 0)
    invalid_pix = ~((pixel_water_count + pixel_land_count) > 0)
    invalid_fmask = None
    invalid_bwtr = None
    if seen_fmask:
        invalid_fmask = ~((fmask_water_count + fmask_land_count) > 0)
    if seen_bwtr:
        invalid_bwtr = ~((bwtr_water_count + bwtr_land_count) > 0)
    invalid_sum = None
    if do_sum and (sum_water_count is not None) and (sum_land_count is not None):
        invalid_sum = ~((sum_water_count + sum_land_count) > 0)

    if write_water_tiffs:
        if out_refl_water_tif is not None:
            out = refl_water_count.copy()
            out[invalid_refl] = nodata_value
            _write_gtiff(out_refl_water_tif, out, gt, ref_wkt, nodata=int(nodata_value))
            written["refl_water"] = out_refl_water_tif
        if out_pixel_water_tif is not None:
            out = pixel_water_count.copy()
            out[invalid_pix] = nodata_value
            _write_gtiff(out_pixel_water_tif, out, gt, ref_wkt, nodata=int(nodata_value))
            written["pixel_water"] = out_pixel_water_tif
        if (out_fmask_water_tif is not None) and seen_fmask and (fmask_water_count is not None):
            out = fmask_water_count.copy()
            out[invalid_fmask] = nodata_value  # type: ignore[index]
            _write_gtiff(out_fmask_water_tif, out, gt, ref_wkt, nodata=int(nodata_value))
            written["fmask_water"] = out_fmask_water_tif
        if (out_bwtr_water_tif is not None) and seen_bwtr and (bwtr_water_count is not None):
            out = bwtr_water_count.copy()
            out[invalid_bwtr] = nodata_value  # type: ignore[index]
            _write_gtiff(out_bwtr_water_tif, out, gt, ref_wkt, nodata=int(nodata_value))
            written["bwtr_water"] = out_bwtr_water_tif
        if do_sum and (sum_water_count is not None) and (out_sum_water_tif is not None) and (invalid_sum is not None):
            out = sum_water_count.copy()
            out[invalid_sum] = nodata_value
            _write_gtiff(out_sum_water_tif, out, gt, ref_wkt, nodata=int(nodata_value))
            written["sum_water"] = out_sum_water_tif

    if write_land_tiffs:
        if out_refl_land_tif is not None:
            out = refl_land_count.copy()
            out[invalid_refl] = nodata_value
            _write_gtiff(out_refl_land_tif, out, gt, ref_wkt, nodata=int(nodata_value))
            written["refl_land"] = out_refl_land_tif
        if out_pixel_land_tif is not None:
            out = pixel_land_count.copy()
            out[invalid_pix] = nodata_value
            _write_gtiff(out_pixel_land_tif, out, gt, ref_wkt, nodata=int(nodata_value))
            written["pixel_land"] = out_pixel_land_tif
        if (out_fmask_land_tif is not None) and seen_fmask and (fmask_land_count is not None):
            out = fmask_land_count.copy()
            out[invalid_fmask] = nodata_value  # type: ignore[index]
            _write_gtiff(out_fmask_land_tif, out, gt, ref_wkt, nodata=int(nodata_value))
            written["fmask_land"] = out_fmask_land_tif
        if (out_bwtr_land_tif is not None) and seen_bwtr and (bwtr_land_count is not None):
            out = bwtr_land_count.copy()
            out[invalid_bwtr] = nodata_value  # type: ignore[index]
            _write_gtiff(out_bwtr_land_tif, out, gt, ref_wkt, nodata=int(nodata_value))
            written["bwtr_land"] = out_bwtr_land_tif
        if do_sum and (sum_land_count is not None) and (out_sum_land_tif is not None) and (invalid_sum is not None):
            out = sum_land_count.copy()
            out[invalid_sum] = nodata_value
            _write_gtiff(out_sum_land_tif, out, gt, ref_wkt, nodata=int(nodata_value))
            written["sum_land"] = out_sum_land_tif

    # Return counts and grid info (valid pixels can be derived as (water+land)>0)
    try:
        feedback.setProgress(100)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText("[BOTH] Done.")
    except Exception:
        pass

    if not seen_fmask:
        fmask_water_count = None
        fmask_land_count = None
    if not seen_bwtr:
        bwtr_water_count = None
        bwtr_land_count = None

    return (
        refl_water_count, refl_land_count,
        pixel_water_count, pixel_land_count,
        fmask_water_count, fmask_land_count,
        bwtr_water_count, bwtr_land_count,
        sum_water_count, sum_land_count,
        gt, ref_wkt, written
    )

def _sum_two_water_masks(
    a_water: np.ndarray,
    a_valid: np.ndarray,
    a_gt,
    a_wkt: str,
    b_water: np.ndarray,
    b_valid: np.ndarray,
    b_gt,
    b_wkt: str,
    out_sum_tif: Path | None,
    out_sum_water_vec: Path | None,
    out_sum_land_vec: Path | None,
    feedback=None,
    smoothify: bool = False,
    smoothify_iters: int = 1,
    smoothify_weight: float = 0.25,
    write_tiffs: bool = False,
    write_water_vec: bool = True,
    write_land_vec: bool = False,
):
    """Sum two water rasters on a common grid.

    If the inputs are per-pixel water *counts* (0..N), the output will be a count
    raster with values 0..(Na+Nb). Vectors are derived from (count > 0).
    """

    s, valid, gt, ref_wkt = _sum_two_masks_to_common_grid(
        a_water, a_valid, a_gt, a_wkt,
        b_water, b_valid, b_gt, b_wkt,
    )

    if write_tiffs and out_sum_tif is not None:
        _write_gtiff(Path(out_sum_tif), s, gt, ref_wkt)

    water_bin = (valid.astype(bool) & (s > 0)).astype(np.uint8)
    land_bin = (valid.astype(bool) & (s == 0)).astype(np.uint8)

    if write_water_vec and out_sum_water_vec is not None:
        ok = _polygonize_binary_array_to_single_feature(
            water_bin,
            gt,
            ref_wkt,
            Path(out_sum_water_vec),
            label="Water",
            class_id=1,
            smoothify=smoothify,
            smoothify_iters=smoothify_iters,
            smoothify_weight=smoothify_weight,
        )
        if not ok:
            feedback.pushInfo("[SUM] No water polygons found; water vector not written.")

    if write_land_vec and out_sum_land_vec is not None:
        ok = _polygonize_binary_array_to_single_feature(
            land_bin,
            gt,
            ref_wkt,
            Path(out_sum_land_vec),
            label="Land",
            class_id=2,
            smoothify=smoothify,
            smoothify_iters=smoothify_iters,
            smoothify_weight=smoothify_weight,
        )
        if not ok:
            feedback.pushInfo("[SUM] No land polygons found; land vector not written.")


def _sum_two_masks_to_common_grid(
    a_water: np.ndarray,
    a_valid: np.ndarray,
    a_gt,
    a_wkt: str,
    b_water: np.ndarray,
    b_valid: np.ndarray,
    b_gt,
    b_wkt: str,
) -> tuple[np.ndarray, np.ndarray, tuple, str]:
    """Return (sum_count, valid_mask, gt, wkt) on a common grid.

    Notes:
      - a_water / b_water may be binary (0/1) OR per-scene water counts (0..N).
      - The returned sum_count is uint16 and represents a per-pixel count.
      - valid_mask is 0/1 (uint8) marking pixels covered by either input.
    """

    # Build 2-band MEM datasets: band1=water_count, band2=valid
    def _mem2(arr1, arr2, gt, wkt):
        h, w = int(arr1.shape[0]), int(arr1.shape[1])
        ds = gdal.GetDriverByName("MEM").Create("", w, h, 2, gdal.GDT_UInt16)
        ds.SetGeoTransform(gt)
        ds.SetProjection(wkt)
        b1 = ds.GetRasterBand(1)
        b2 = ds.GetRasterBand(2)
        b1.WriteArray(np.clip(arr1, 0, 65535).astype(np.uint16))
        b2.WriteArray(np.clip(arr2, 0, 1).astype(np.uint16))
        b1.SetNoDataValue(0)
        b2.SetNoDataValue(0)
        b1.FlushCache(); b2.FlushCache()
        return ds

    ds_a = _mem2(a_water, a_valid, a_gt, a_wkt)
    ds_b = _mem2(b_water, b_valid, b_gt, b_wkt)

    # Choose reference SRS from A (they should be projected consistently if same path/row)
    ref_wkt = a_wkt
    if not ref_wkt:
        raise ValueError("SUM: reference raster has no CRS.")
    ref_srs = _srs_from_wkt(ref_wkt)

    agt = ds_a.GetGeoTransform()
    bgt = ds_b.GetGeoTransform()
    xres = min(abs(agt[1]), abs(bgt[1]))
    yres = min(abs(agt[5]), abs(bgt[5]))

    aminx, aminy, amaxx, amaxy = _densified_bounds_in_target(ds_a, ref_srs, densify=21)
    bminx, bminy, bmaxx, bmaxy = _densified_bounds_in_target(ds_b, ref_srs, densify=21)
    minx = min(aminx, bminx)
    miny = min(aminy, bminy)
    maxx = max(amaxx, bmaxx)
    maxy = max(amaxy, bmaxy)

    # Snap relative to the reference raster origin to avoid half-pixel shifts.
    left, bottom, right, top = _snap_bounds(
        minx, miny, maxx, maxy, xres, yres,
        anchor_x=float(agt[0]),
        anchor_y=float(agt[3]),
    )
    width = int(math.ceil((right - left) / xres))
    height = int(math.ceil((top - bottom) / yres))
    bounds = (left, bottom, right, top)
    gt = (left, xres, 0.0, top, 0.0, -yres)

    # Warp both bands to common grid (nearest)
    wa = _warp_to_grid(ds_a, ref_wkt, bounds, xres, yres, width, height, dst_alpha=False, out_dtype=gdal.GDT_UInt16)
    wb = _warp_to_grid(ds_b, ref_wkt, bounds, xres, yres, width, height, dst_alpha=False, out_dtype=gdal.GDT_UInt16)

    wa_water = wa.GetRasterBand(1).ReadAsArray().astype(np.uint16)
    wa_valid = wa.GetRasterBand(2).ReadAsArray().astype(np.uint16)
    wb_water = wb.GetRasterBand(1).ReadAsArray().astype(np.uint16)
    wb_valid = wb.GetRasterBand(2).ReadAsArray().astype(np.uint16)

    valid = (wa_valid > 0) | (wb_valid > 0)
    s = np.clip(wa_water + wb_water, 0, 65535).astype(np.uint16)

    return s, valid.astype(np.uint8), gt, ref_wkt


class SentinelWaterMaskAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing algorithm.

    - Inputs: raster layers currently loaded in the project (or optionally, a selected subset)
    - Outputs: rasters/vectors returned to QGIS as Processing destinations (TEMPORARY_OUTPUT by default)
    """

    INPUT_LAYERS = "INPUT_LAYERS"
    MODE = "MODE"

    SEG_MONTHS = "SEG_MONTHS"
    SEG_YEARS = "SEG_YEARS"
    SEG_OUTPUT_DIR = "SEG_OUTPUT_DIR"

    BG_RGB = "BG_RGB"
    KEEP_DEFAULT = "KEEP_DEFAULT"
    RGB_THRESHOLDS = "RGB_THRESHOLDS"

    PIXEL_KEEP_DEFAULT_WATER = "PIXEL_KEEP_DEFAULT_WATER"
    PIXEL_WATER_VALUES_457 = "PIXEL_WATER_VALUES_457"
    PIXEL_WATER_VALUES_89 = "PIXEL_WATER_VALUES_89"

    SENTINEL_WATER_MODE = "SENTINEL_WATER_MODE"
    SENTINEL_WATER_VALUES_CUSTOM = "SENTINEL_WATER_VALUES_CUSTOM"

    # Sentinel integration controls (Guided UI exposes these; Processing dialog keeps them Advanced)
    USE_SENTINEL = "USE_SENTINEL"
    SENTINEL_SUM_WITH_LANDSAT = "SENTINEL_SUM_WITH_LANDSAT"

    # OPERA integration controls (DSWx-HLS BWTR)
    USE_OPERA = "USE_OPERA"
    OPERA_SUM_WITH_LANDSAT = "OPERA_SUM_WITH_LANDSAT"

    # If running in PIXEL mode with only augmentations (FMASK + BWTR), optionally
    # produce SUM outputs from those two sources (no Landsat REFL/QA_PIXEL required).
    DO_FMASK_BWTR_SUM = "DO_FMASK_BWTR_SUM"

    PIXEL_SMOOTH = "PIXEL_SMOOTH"
    PIXEL_SMOOTH_SIZE = "PIXEL_SMOOTH_SIZE"

    SMOOTHIFY = "SMOOTHIFY"
    SMOOTHIFY_ITERS = "SMOOTHIFY_ITERS"
    SMOOTHIFY_WEIGHT = "SMOOTHIFY_WEIGHT"
    SMOOTHIFY_PRESIMPLIFY_M = "SMOOTHIFY_PRESIMPLIFY_M"
    DO_SUM = "DO_SUM"

    WRITE_TIFFS = "WRITE_TIFFS"
    WRITE_LAND_TIFFS = "WRITE_LAND_TIFFS"
    VEC_WRITE = "VEC_WRITE"

    OUT_REFL_TIF = "OUT_REFL_TIF"
    OUT_REFL_LAND_TIF = "OUT_REFL_LAND_TIF"
    OUT_REFL_VEC = "OUT_REFL_VEC"
    OUT_REFL_LAND_VEC = "OUT_REFL_LAND_VEC"
    OUT_PIXEL_TIF = "OUT_PIXEL_TIF"
    OUT_PIXEL_LAND_TIF = "OUT_PIXEL_LAND_TIF"
    OUT_PIXEL_VEC = "OUT_PIXEL_VEC"
    OUT_PIXEL_LAND_VEC = "OUT_PIXEL_LAND_VEC"
    OUT_FMASK_TIF = "OUT_FMASK_TIF"
    OUT_FMASK_LAND_TIF = "OUT_FMASK_LAND_TIF"
    OUT_FMASK_VEC = "OUT_FMASK_VEC"
    OUT_FMASK_LAND_VEC = "OUT_FMASK_LAND_VEC"
    OUT_BWTR_TIF = "OUT_BWTR_TIF"
    OUT_BWTR_LAND_TIF = "OUT_BWTR_LAND_TIF"
    OUT_BWTR_VEC = "OUT_BWTR_VEC"
    OUT_BWTR_LAND_VEC = "OUT_BWTR_LAND_VEC"
    OUT_SUM_TIF = "OUT_SUM_TIF"
    OUT_SUM_LAND_TIF = "OUT_SUM_LAND_TIF"
    OUT_SUM_VEC = "OUT_SUM_VEC"
    OUT_SUM_LAND_VEC = "OUT_SUM_LAND_VEC"

    # Newline-separated lists of all written TIFFs (when multiple Path/Row groups exist)
    OUT_REFL_TIF_LIST = "OUT_REFL_TIF_LIST"
    OUT_REFL_LAND_TIF_LIST = "OUT_REFL_LAND_TIF_LIST"
    OUT_PIXEL_TIF_LIST = "OUT_PIXEL_TIF_LIST"
    OUT_PIXEL_LAND_TIF_LIST = "OUT_PIXEL_LAND_TIF_LIST"
    OUT_FMASK_TIF_LIST = "OUT_FMASK_TIF_LIST"
    OUT_FMASK_LAND_TIF_LIST = "OUT_FMASK_LAND_TIF_LIST"
    OUT_BWTR_TIF_LIST = "OUT_BWTR_TIF_LIST"
    OUT_BWTR_LAND_TIF_LIST = "OUT_BWTR_LAND_TIF_LIST"
    OUT_SUM_TIF_LIST = "OUT_SUM_TIF_LIST"
    OUT_SUM_LAND_TIF_LIST = "OUT_SUM_LAND_TIF_LIST"

    OUT_LOG = "OUT_LOG"

    def flags(self):
        return super().flags() | QgsProcessingAlgorithm.FlagHideFromToolbox

    def name(self):
        return "sentinel_water_mask"

    def displayName(self):
        return "Landsat Water Mask (Landsat QA_PIXEL + HLS S30 Fmask + OPERA DSWx-HLS)"

    def group(self):
        return "Landsat Water Mask"

    def groupId(self):
        return "sentinel_water_mask"

    def shortHelpString(self):
        return (
            "Builds water count masks (0..N, where N is the number of scenes) and dissolved water polygons from Landsat rasters.\n\n"
            "Acceptable file inputs can be sourced from Earth Explorer.\n"
            "1. Landsat Collection 2 Level-2 ⇒ Landsat 8-9 OLI/TIRS C2 L2 ⇒ Landsat 7 ETM+ C2 L2 ⇒ Landsat 4-5 TM C2L2 ↪ QA_PIXEL.TIF\n"
            "2. Landsat Collection 2 Level-1 ⇒ Landsat 8-9 OLI/TIRS C2 L1 ⇒ Landsat 7 ETM+ C2 L1 ⇒ Landsat 4-5 TM C2L1 ↪ Full Resolution Browse (Reflective Color) GeoTIFF\n"
            "REFL: counts water per pixel across one/more *_refl.tif using inclusive RGB ranges (R,G,B).\n"
            "PIXEL: counts water per pixel across one/more QA_PIXEL rasters using QA_PIXEL equality (water value depends on Landsat generation).\n"
            "Landsat generation # detected from filename characters 3 – 4 (04, 05, 07, 08, 09):\n"
            "• Landsat 4/5/7: water == 5504\n"
            "• Landsat 8/9:   water == 21952\n"
            "BOTH: runs both file types and can optionally write a water_sum raster and polygons.\n\n"
            "Outputs are Processing destinations (TEMPORARY_OUTPUT by default) and will be added back into QGIS.\n"
            "Path/Row support: the tool groups inputs by Landsat WRS-2 Path/Row found in the filename token '_PPPRRR_'.\n"
            "Example: LC09_L2SP_022039_20251211_... => Path=022, Row=039. Each group is processed independently, then final vectors are merged + dissolved.\n\n"
            "Smoothing: choose ONE — Pixel smoothing (fast) smooths the binary mask and applies a light vector smoothing; Smoothify (more intensive) smooths the final merged vectors (Chaikin)."
        )

    def createInstance(self):
        return SentinelWaterMaskAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT_LAYERS,
                "Input raster layers (optional). Leave empty to auto-use ALL raster layers loaded in the current QGIS project.",
                layerType=QgsProcessing.TypeRaster,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE,
                "Mode",
                options=["REFL", "PIXEL", "BOTH"],
                defaultValue=2
            )
        )


        # --- Optional segmentation (runs the same logic multiple times) ---
        # Month segments examples:
        #   Jan-Mar;Oct-Dec
        #   1-3;10-12
        #   Nov-Feb  (wrap range)
        self.addParameter(
            QgsProcessingParameterString(
                self.SEG_MONTHS,
                "Segmentation: month ranges (optional; ';' separated)",
                defaultValue="",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.SEG_YEARS,
                "Segmentation: year ranges (optional; ';' separated)",
                defaultValue="",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.SEG_OUTPUT_DIR,
                "Segmentation: output folder (optional; required to persist multiple outputs)",
                optional=True,
            )
        )
        # --- Output selection ---
        self.addParameter(
            QgsProcessingParameterEnum(
                self.VEC_WRITE,
                "Write shapefiles",
                options=["Water", "Land", "Water + Land"],
                defaultValue=2
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WRITE_TIFFS,
                "Write Water Classification Count TIFF rasters (optional)",
                defaultValue=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WRITE_LAND_TIFFS,
                "Write Land Classification Count TIFF rasters (optional)",
                defaultValue=False,
            )
        )


        # --- REFL-only options (kept in the Advanced section to keep PIXEL mode clean) ---
        p_bg = QgsProcessingParameterString(
            self.BG_RGB,
            "REFL: background RGB to exclude (r,g,b)",
            defaultValue="0,0,0",
        )
        p_bg.setFlags(p_bg.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_bg)

        p_def = QgsProcessingParameterBoolean(
            self.KEEP_DEFAULT,
            "REFL: use default water RGB thresholds (recommended)",
            defaultValue=True,
        )
        p_def.setFlags(p_def.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_def)

        # Matrix is a single row of 6 values: [Rmin, Rmax, Gmin, Gmax, Bmin, Bmax]
        # This keeps the dialog compact versus six separate numeric controls.
        if QgsProcessingParameterMatrix is not None:
            p_rgb = QgsProcessingParameterMatrix(
                self.RGB_THRESHOLDS,
                "REFL: custom RGB thresholds (Rmin,Rmax,Gmin,Gmax,Bmin,Bmax)\n(used only when 'use default' is unchecked)",
                numberRows=1,
                hasFixedNumberRows=True,
                defaultValue=[0, 49, 0, 49, 11, 255],
                headers=["R min", "R max", "G min", "G max", "B min", "B max"],
            )
            p_rgb.setFlags(p_rgb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(p_rgb)
        else:
            # QGIS <= 3.10 doesn't have QgsProcessingParameterMatrix. Use a compact string instead.
            p_rgb = QgsProcessingParameterString(
                self.RGB_THRESHOLDS,
                "REFL: custom RGB thresholds (Rmin,Rmax,Gmin,Gmax,Bmin,Bmax)\n(used only when 'use default' is unchecked)",
                defaultValue="0,49,0,49,11,255",
            )
            p_rgb.setFlags(p_rgb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(p_rgb)


        
        # --- PIXEL water code options (Advanced) ---
        p_pwdef = QgsProcessingParameterBoolean(
            self.PIXEL_KEEP_DEFAULT_WATER,
            "PIXEL: use default water codes (recommended)",
            defaultValue=True,
        )
        p_pwdef.setFlags(p_pwdef.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_pwdef)

        p_p457 = QgsProcessingParameterString(
            self.PIXEL_WATER_VALUES_457,
            "PIXEL: water codes for Landsat 4/5/7 (comma-separated)",
            defaultValue="5504",
        )
        p_p457.setFlags(p_p457.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_p457)

        p_p89 = QgsProcessingParameterString(
            self.PIXEL_WATER_VALUES_89,
            "PIXEL: water codes for Landsat 8/9 (comma-separated)",
            defaultValue="21952",
        )
        p_p89.setFlags(p_p89.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_p89)

        # --- Sentinel-2 HLS Fmask water code options (Advanced) ---
        p_swmode = QgsProcessingParameterEnum(
            self.SENTINEL_WATER_MODE,
            "SENTINEL Fmask: water category",
            options=["All Water", "Pure Water", "Custom"],
            defaultValue=0,
        )
        p_swmode.setFlags(p_swmode.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_swmode)

        p_swvals = QgsProcessingParameterString(
            self.SENTINEL_WATER_VALUES_CUSTOM,
            "SENTINEL Fmask: custom water values (comma-separated, 0–255)\n(used only when category is 'Custom')",
            defaultValue="32,96,160,224",
        )
        p_swvals.setFlags(p_swvals.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_swvals)

        # --- Sentinel integration toggles (Advanced in Processing; primary controls in Guided UI) ---
        p_use_sentinel = QgsProcessingParameterBoolean(
            self.USE_SENTINEL,
            "Use Sentinel-2 HLS S30 Fmask layers (augment/merge)",
            defaultValue=False,
        )
        p_use_sentinel.setFlags(p_use_sentinel.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_use_sentinel)

        p_sum_sentinel = QgsProcessingParameterBoolean(
            self.SENTINEL_SUM_WITH_LANDSAT,
            "If Sentinel is enabled, also include it in SUM outputs (where applicable)",
            defaultValue=True,
        )
        p_sum_sentinel.setFlags(p_sum_sentinel.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sum_sentinel)

        # --- OPERA DSWx-HLS integration toggles (Advanced in Processing; primary controls in Guided UI) ---
        p_use_opera = QgsProcessingParameterBoolean(
            self.USE_OPERA,
            "Use OPERA L3 DSWx-HLS BWTR layers (augment/merge)",
            defaultValue=False,
        )
        p_use_opera.setFlags(p_use_opera.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_use_opera)

        p_sum_opera = QgsProcessingParameterBoolean(
            self.OPERA_SUM_WITH_LANDSAT,
            "If OPERA is enabled, also include it in SUM outputs (where applicable)",
            defaultValue=True,
        )
        p_sum_opera.setFlags(p_sum_opera.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sum_opera)

# --- Fast pixel-based smoothing (recommended for large rasters) ---
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PIXEL_SMOOTH,
                'Pixel smoothing (fast; majority filter on mask before polygonize)',
                defaultValue=False,
            )
        )

        p_pk = QgsProcessingParameterNumber(
            self.PIXEL_SMOOTH_SIZE,
            'Pixel smoothing kernel size (odd pixels; 3 recommended)',
            QgsProcessingParameterNumber.Integer,
            3,
            minValue=1,
            maxValue=31,
        )
        p_pk.setFlags(p_pk.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_pk)

        # --- Output smoothing ---
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SMOOTHIFY,
                "Smoothify output polygons",
                defaultValue=False,
            )
        )

        p_it = QgsProcessingParameterNumber(
            self.SMOOTHIFY_ITERS,
            "Smoothify iterations (Chaikin)",
            QgsProcessingParameterNumber.Integer,
            1,
            minValue=1,
            maxValue=8,
        )
        p_it.setFlags(p_it.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_it)

        p_w = QgsProcessingParameterNumber(
            self.SMOOTHIFY_WEIGHT,
            "Smoothify weight (0..0.5)",
            QgsProcessingParameterNumber.Double,
            0.25,
            minValue=0.05,
            maxValue=0.49,
        )
        p_w.setFlags(p_w.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_w)

        p_ps = QgsProcessingParameterNumber(
            self.SMOOTHIFY_PRESIMPLIFY_M,
            "Smoothify pre-simplify tolerance (meters; 0 disables)",
            QgsProcessingParameterNumber.Double,
            15.0,
            minValue=0.0,
            maxValue=1000.0,
        )
        p_ps.setFlags(p_ps.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ps)

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DO_SUM,
                "BOTH: also write water_sum outputs (requires both REFL and QA_PIXEL layers)",
                defaultValue=True,
            )
        )

        p_aug_sum = QgsProcessingParameterBoolean(
            self.DO_FMASK_BWTR_SUM,
            "PIXEL: also write SUM outputs from FMASK + BWTR (no Landsat required)",
            defaultValue=False,
        )
        p_aug_sum.setFlags(p_aug_sum.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_aug_sum)

        # --- Outputs (default to temporary; QGIS will add them to the map) ---

        # Note: TIFF destinations are only used when "Write output TIFF masks" is enabled.
        # They are marked as Advanced to keep the dialog clean for vector-only workflows.
        p_refl_tif = QgsProcessingParameterRasterDestination(
            self.OUT_REFL_TIF,
            "REFL water count raster (0..Ndates) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_refl_tif.setFlags(p_refl_tif.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_refl_tif)

        p_refl_land_tif = QgsProcessingParameterRasterDestination(
            self.OUT_REFL_LAND_TIF,
            "REFL land count raster (0..Ndates) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_refl_land_tif.setFlags(p_refl_land_tif.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_refl_land_tif)

        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUT_REFL_VEC,
                "REFL water polygons (EPSG:4326)",
                defaultValue="TEMPORARY_OUTPUT",
            )
        )
        p_refl_land = QgsProcessingParameterVectorDestination(
            self.OUT_REFL_LAND_VEC,
            "REFL land polygons (EPSG:4326) [non-water, excluding NoData/background]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_refl_land.setFlags(p_refl_land.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_refl_land)

        p_pixel_tif = QgsProcessingParameterRasterDestination(
            self.OUT_PIXEL_TIF,
            "QA_PIXEL water count raster (0..Ndates) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_pixel_tif.setFlags(p_pixel_tif.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_pixel_tif)

        p_pixel_land_tif = QgsProcessingParameterRasterDestination(
            self.OUT_PIXEL_LAND_TIF,
            "QA_PIXEL land count raster (0..Ndates) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_pixel_land_tif.setFlags(p_pixel_land_tif.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_pixel_land_tif)

        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUT_PIXEL_VEC,
                "QA_PIXEL water polygons (EPSG:4326)",
                defaultValue="TEMPORARY_OUTPUT",
            )
        )
        p_pixel_land = QgsProcessingParameterVectorDestination(
            self.OUT_PIXEL_LAND_VEC,
            "QA_PIXEL land polygons (EPSG:4326) [non-water, excluding NoData/background]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_pixel_land.setFlags(p_pixel_land.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_pixel_land)

        # --- Sentinel HLS S30 Fmask outputs (byte categorical) ---
        p_fmask_tif = QgsProcessingParameterRasterDestination(
            self.OUT_FMASK_TIF,
            "FMASK water count raster (0..Ndates) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_fmask_tif.setFlags(p_fmask_tif.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_fmask_tif)

        p_fmask_land_tif = QgsProcessingParameterRasterDestination(
            self.OUT_FMASK_LAND_TIF,
            "FMASK land count raster (0..Ndates) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_fmask_land_tif.setFlags(p_fmask_land_tif.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_fmask_land_tif)

        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUT_FMASK_VEC,
                "FMASK water polygons (EPSG:4326) [optional]",
                defaultValue="TEMPORARY_OUTPUT",
            )
        )
        p_fmask_land = QgsProcessingParameterVectorDestination(
            self.OUT_FMASK_LAND_VEC,
            "FMASK land polygons (EPSG:4326) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_fmask_land.setFlags(p_fmask_land.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_fmask_land)

        # --- OPERA DSWx-HLS BWTR outputs (byte water mask) ---
        p_bwtr_tif = QgsProcessingParameterRasterDestination(
            self.OUT_BWTR_TIF,
            "BWTR water count raster (0..Ndates) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_bwtr_tif.setFlags(p_bwtr_tif.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_bwtr_tif)

        p_bwtr_land_tif = QgsProcessingParameterRasterDestination(
            self.OUT_BWTR_LAND_TIF,
            "BWTR land count raster (0..Ndates) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_bwtr_land_tif.setFlags(p_bwtr_land_tif.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_bwtr_land_tif)

        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUT_BWTR_VEC,
                "BWTR water polygons (EPSG:4326) [optional]",
                defaultValue="TEMPORARY_OUTPUT",
            )
        )
        p_bwtr_land = QgsProcessingParameterVectorDestination(
            self.OUT_BWTR_LAND_VEC,
            "BWTR land polygons (EPSG:4326) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_bwtr_land.setFlags(p_bwtr_land.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_bwtr_land)

        p_sum_tif = QgsProcessingParameterRasterDestination(
            self.OUT_SUM_TIF,
            "SUM water count raster (REFL+QA, 0..Ndates) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_sum_tif.setFlags(p_sum_tif.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sum_tif)

        p_sum_land_tif = QgsProcessingParameterRasterDestination(
            self.OUT_SUM_LAND_TIF,
            "SUM land count raster (REFL+QA, 0..Ndates) [optional]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_sum_land_tif.setFlags(p_sum_land_tif.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sum_land_tif)

        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUT_SUM_VEC,
                "SUM water polygons (EPSG:4326) [built from SUM count raster]",
                defaultValue="TEMPORARY_OUTPUT",
            )
        )
        p_sum_land = QgsProcessingParameterVectorDestination(
            self.OUT_SUM_LAND_VEC,
            "SUM land polygons (EPSG:4326) [inverse of SUM water within valid pixels; excluding NoData/background]",
            defaultValue="TEMPORARY_OUTPUT",
        )
        p_sum_land.setFlags(p_sum_land.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sum_land)

        if QgsProcessingOutputString is not None:
            # Helpful for scripts/GUI: Processing destinations only return one path per raster output;
            # these strings contain ALL written rasters when multiple Path/Row groups exist.
            self.addOutput(QgsProcessingOutputString(self.OUT_REFL_TIF_LIST, "REFL Water Classification Count TIFF paths (newline-separated)"))
            self.addOutput(QgsProcessingOutputString(self.OUT_REFL_LAND_TIF_LIST, "REFL Land Classification Count TIFF paths (newline-separated)"))
            self.addOutput(QgsProcessingOutputString(self.OUT_PIXEL_TIF_LIST, "QA_PIXEL Water Classification Count TIFF paths (newline-separated)"))
            self.addOutput(QgsProcessingOutputString(self.OUT_PIXEL_LAND_TIF_LIST, "QA_PIXEL Land Classification Count TIFF paths (newline-separated)"))
            self.addOutput(QgsProcessingOutputString(self.OUT_SUM_TIF_LIST, "SUM Water Classification Count TIFF paths (newline-separated)"))
            self.addOutput(QgsProcessingOutputString(self.OUT_SUM_LAND_TIF_LIST, "SUM Land Classification Count TIFF paths (newline-separated)"))
            self.addOutput(QgsProcessingOutputString(self.OUT_LOG, "Log"))

    def _collect_layers(self, parameters, context):
        """Collect input layers.

        IMPORTANT: This algorithm is designed to be safe when run as a background
        task. Therefore it must NOT access QgsProject.instance() or any map layers
        owned by the main thread.

        The guided UI supplies the list of layers (or file paths). When run from
        the Processing Toolbox, users must explicitly set INPUT_LAYERS.
        """

        layers = self.parameterAsLayerList(parameters, self.INPUT_LAYERS, context) or []
        if not layers:
            raise QgsProcessingException(
                "No input layers were provided. Please set INPUT_LAYERS (or use the guided dialog)."
            )
        return layers

    def _categorize(self, layers):
        """Split inputs into REFL, Landsat QA_PIXEL, Sentinel HLS S30 Fmask, and OPERA DSWx-HLS BWTR."""
        refl_paths = []
        qa_pixel_paths = []
        sentinel_paths = []
        opera_paths = []

        for lyr in layers:
            # layers may be QgsRasterLayer (typical) or a string/path (some QGIS versions)
            if isinstance(lyr, QgsRasterLayer):
                try:
                    src = (lyr.source() or "").split("|")[0]
                except Exception:
                    src = ""
                try:
                    name_l = (lyr.name() or "").lower()
                except Exception:
                    name_l = ""
            else:
                src = str(lyr)
                name_l = ""

            if not src:
                continue

            p = Path(src)
            fname_l = p.name.lower()

            if "_refl" in fname_l or "_refl" in name_l:
                refl_paths.append(p)
            if "qa_pixel" in fname_l or "qa_pixel" in name_l:
                qa_pixel_paths.append(p)
            # Sentinel-2 HLS S30 Fmask (byte)
            if "fmask" in fname_l and ("hls.s30" in fname_l or fname_l.startswith("hls.s30")):
                sentinel_paths.append(p)

            # OPERA L3 DSWx-HLS BWTR (byte; water==1; nodata==255)
            if _is_opera_dswx_name(p.name):
                opera_paths.append(p)

        return _dedupe_paths(refl_paths), _dedupe_paths(qa_pixel_paths), _dedupe_paths(sentinel_paths), _dedupe_paths(opera_paths)
    
    def processAlgorithm(self, parameters, context, feedback):
            mode_idx = self.parameterAsEnum(parameters, self.MODE, context)
            mode = ["refl", "pixel", "both"][mode_idx]

            vec_idx = self.parameterAsEnum(parameters, self.VEC_WRITE, context)
            vec_mode = ["water", "land", "both"][vec_idx]
            write_water_vec = vec_mode in ("water", "both")
            write_land_vec = vec_mode in ("land", "both")
            write_water_tiffs = self.parameterAsBool(parameters, self.WRITE_TIFFS, context)
            write_land_tiffs = self.parameterAsBool(parameters, self.WRITE_LAND_TIFFS, context)
            write_any_tiffs = write_water_tiffs or write_land_tiffs

            all_layers = self._collect_layers(parameters, context)

            # Segmentation: run the same core logic multiple times, each time on a date-filtered subset.
            month_spec = self.parameterAsString(parameters, self.SEG_MONTHS, context)
            year_spec = self.parameterAsString(parameters, self.SEG_YEARS, context)
            seg_months = parse_month_segments(month_spec)
            seg_years = parse_year_segments(year_spec)

            seg_enabled = bool(seg_months or seg_years)

            if not seg_enabled:
                return self._process_core(parameters, context, feedback, all_layers)

            # If user didn't set an output folder, use a temp folder (still returns paths in output strings).
            out_dir = self.parameterAsString(parameters, self.SEG_OUTPUT_DIR, context) or ""
            if not out_dir:
                import tempfile
                out_dir = tempfile.mkdtemp(prefix="sentinel_watermask_segments_")
            out_dir_p = Path(out_dir)
            out_dir_p.mkdir(parents=True, exist_ok=True)

            # Default to "all" if one dimension omitted
            if not seg_months:
                seg_months = [{"start": 1, "end": 12, "wrap": False, "months": set(range(1, 13)), "label": "AllMonths"}]
            if not seg_years:
                seg_years = [{"start": -10**9, "end": 10**9, "label": "AllYears"}]

            # Pre-extract acquisition date from each layer's source/name (expects YYYYMMDD or YYYYDOYThhmmss in filename).
            layer_info = []
            for lyr in all_layers:
                try:
                    src = lyr.source()
                except Exception:
                    src = str(lyr)
                fname = Path(src).name
                acq_date = _parse_acq_date_from_string(fname)
                if not acq_date:
                    # also try layer name as a fallback
                    try:
                        acq_date = _parse_acq_date_from_string(lyr.name() or "")
                    except Exception:
                        acq_date = None
                if not acq_date:
                    feedback.pushInfo(f"[SEG] Skipping (no recognizable date token found; expected YYYYMMDD or YYYYDOYThhmmss (e.g., 2026001T163711)): {fname}")
                    continue
                # We already parsed a concrete date (either YYYYMMDD or YYYYDOYThhmmss) into acq_date
                layer_info.append((lyr, acq_date.year, acq_date.month))

            if not layer_info:
                raise QgsProcessingException("Segmentation is enabled, but none of the input layers contain an acquisition date token like _YYYYMMDD_ or .YYYYDOYThhmmss.")

            # Build a compact tag describing which inputs are used for outputs
            try:
                _refl_all, _qa_all, _sent_all, _opera_all = self._categorize(all_layers)
            except Exception:
                _refl_all, _qa_all, _sent_all, _opera_all = ([], [], [], [])
            tag_parts = []
            if mode in ("pixel", "both") and _qa_all:
                tag_parts.append("PIXEL")
            if mode in ("refl", "both") and _refl_all:
                tag_parts.append("REFL")
            if _sent_all:
                tag_parts.append("FMASK")
            if '_opera_all' in locals() and _opera_all:
                tag_parts.append("BWTR")
            _SWM_DATA_TAG = "_".join(tag_parts) if tag_parts else "DATA"
            globals()["_SWM_DATA_TAG"] = _SWM_DATA_TAG

            # Run combinations (cartesian product), but precompute work so we can show a meaningful
            # progress bar based on completed *input files* (layers) rather than segments.
            combos = [(yseg, mseg) for yseg in seg_years for mseg in seg_months]

            # Precompute which layers fall into each segment and the total work (layer-count).
            combos_with_layers = []  # list[(yseg, mseg, [layers])]
            total_files = 0
            for (yseg, mseg) in combos:
                sub_layers = [lyr for (lyr, y, m) in layer_info
                              if (yseg["start"] <= y <= yseg["end"]) and (m in mseg["months"])]
                if not sub_layers:
                    continue
                combos_with_layers.append((yseg, mseg, sub_layers))
                total_files += len(sub_layers)

            n = len(combos_with_layers)
            done_files = 0

            try:
                feedback.setProgress(0)
                if hasattr(feedback, "setProgressText"):
                    feedback.setProgressText(f"[SEG] 0/{total_files} files complete ({n} segments)")
            except Exception:
                pass
            all_refl_tif = []
            all_refl_land_tif = []
            all_pix_tif = []
            all_pix_land_tif = []
            all_sum_tif = []
            all_sum_land_tif = []
            log_lines = [f"Segmentation enabled. Output folder: {out_dir_p}",
                         f"Year segments: {';'.join([s['label'] for s in seg_years])}",
                         f"Month segments: {';'.join([s['label'] for s in seg_months])}",
                         ""]

            first_primary_results = None

            class _ScaledFeedback:
                """Scale core progress (0-100) into global segmentation progress.

                QGIS core progress is usually per-segment and very chatty; we want the overall
                bar to reflect completed *input files* across all segments.
                """

                def __init__(self, parent_fb, done_files0: int, seg_size: int, total_files_: int):
                    self._fb = parent_fb
                    self._done0 = max(0, int(done_files0))
                    self._seg = max(1, int(seg_size))
                    self._total = max(1, int(total_files_))

                def isCanceled(self):
                    return self._fb.isCanceled()

                def pushInfo(self, msg):
                    return self._fb.pushInfo(msg)

                def pushWarning(self, msg):
                    return getattr(self._fb, "pushWarning", self._fb.pushInfo)(msg)

                def reportError(self, msg, fatalError=False):
                    return getattr(self._fb, "reportError", self._fb.pushInfo)(msg)

                def setProgress(self, v):
                    try:
                        frac = max(0.0, min(1.0, float(v) / 100.0))
                    except Exception:
                        frac = 0.0
                    done = self._done0 + int(round(frac * self._seg))
                    pct = int(round(100.0 * done / self._total))
                    try:
                        self._fb.setProgress(pct)
                    except Exception:
                        pass

                def setProgressText(self, t):
                    # Suppress chatty core messages from overwriting the segmentation header.
                    return None

                def setProgressTextAndProgress(self, t, v):
                    self.setProgress(v)
                    return None

                def __getattr__(self, name):
                    return getattr(self._fb, name)

            for i, (yseg, mseg, sub_layers) in enumerate(combos_with_layers):
                if feedback.isCanceled():
                    break
                y_label = yseg["label"]
                m_label = mseg["label"]
                suffix = _safe_suffix(f"Y{y_label}_M{m_label}")

                # sub_layers is precomputed

                # Build per-segment parameters.
                # NOTE: do this ONCE. (A previous duplicate dict() assignment could silently
                # wipe per-segment settings.)
                p2 = dict(parameters)

                # In segmented mode, honor the user's TIFF output settings.

                p2["_SWM_SEGMENTED"] = True

                # Build a tag describing which inputs contributed (QA/REFL/FMACKS)
                data_tag = globals().get("_SWM_DATA_TAG", None)
                if not data_tag:
                    data_tag = "DATA"

                def _year_file_label(seg: dict) -> str:
                    try:
                        s, e = int(seg.get("start")), int(seg.get("end"))
                        if s <= -10**8 or e >= 10**8:
                            return "AllYears"
                        if s == e:
                            return f"{s}"
                        return f"{s}_{e}"
                    except Exception:
                        return (seg.get("label") or "AllYears").replace("-", "_").replace(" ", "")

                def _month_file_label(seg: dict) -> str:
                    return (seg.get("label") or "AllMonths").replace("-", "_").replace(" ", "")

                # Build per-segment token used for folder + filenames (e.g., "2024_2025_AllMonths")
                try:
                    seg_token = _safe_suffix(f"{_year_file_label(yseg)}_{_month_file_label(mseg)}")
                except Exception:
                    seg_token = _safe_suffix(f"{y_label}_{m_label}")

                # Desired filenames and per-segment folder (write *all* outputs just like non-segmented runs)
                # Create a subfolder inside the chosen output directory for each segment so results stay organized.
                seg_folder = out_dir_p / seg_token
                try:
                    seg_folder.mkdir(parents=True, exist_ok=True)
                except Exception:
                    # If we can't create the folder for any reason, fall back to the base output dir.
                    seg_folder = out_dir_p

                # Helper: absolute path in the segment folder
                def _p(name: str) -> str:
                    return str(seg_folder / name)

                # In segmentation runs, keep ALL official output parameters empty so QGIS will never
                # auto-load outputs into the project. We still write *real* files to disk using internal
                # destination overrides.
                #
                # Important: clear *all* outputs up-front (not just the ones we override). Otherwise,
                # outputs left as TEMPORARY_OUTPUT by Processing can sometimes be materialized and loaded.
                def _seg_dest_key(k: str) -> str:
                    return f"_SWM_SEG_DEST_{k}"

                # Clear all official outputs (prevents any accidental TEMPORARY_OUTPUT layers)
                for _k in (
                    self.OUT_REFL_TIF, self.OUT_REFL_LAND_TIF, self.OUT_REFL_VEC, self.OUT_REFL_LAND_VEC,
                    self.OUT_PIXEL_TIF, self.OUT_PIXEL_LAND_TIF, self.OUT_PIXEL_VEC, self.OUT_PIXEL_LAND_VEC,
                    self.OUT_FMASK_TIF, self.OUT_FMASK_LAND_TIF, self.OUT_FMASK_VEC, self.OUT_FMASK_LAND_VEC,
                    self.OUT_BWTR_TIF, self.OUT_BWTR_LAND_TIF, self.OUT_BWTR_VEC, self.OUT_BWTR_LAND_VEC,
                    self.OUT_SUM_TIF, self.OUT_SUM_LAND_TIF, self.OUT_SUM_VEC, self.OUT_SUM_LAND_VEC,
                    self.OUT_LOG,
                ):
                    try:
                        p2[_k] = ""
                    except Exception:
                        pass

                def _set_seg_dest(k: str, path: str):
                    p2[_seg_dest_key(k)] = path
                    # Clear official output so QGIS doesn't auto-load
                    p2[k] = ""

# Build the SUM label from sources that are ACTUALLY present in this segment
                # (and enabled by the UI). This keeps filenames honest when a segment contains, e.g.,
                # REFL+PIXEL+BWTR but no FMASK.
                seg_refl, seg_qa, seg_sent, seg_opera = self._categorize(sub_layers)
                # Respect guided-UI flags for segmentation naming too.
                if not p2.get('USE_REFL', True):
                    seg_refl = []
                if not p2.get('USE_QA_PIXEL', True):
                    seg_qa = []
                if not p2.get('USE_FMASK', True):
                    seg_sent = []
                if not p2.get('USE_BWTR', True):
                    seg_opera = []

                sel_sources = []
                if seg_refl:
                    sel_sources.append("REFL")
                if seg_qa:
                    sel_sources.append("PIXEL")
                if seg_sent:
                    sel_sources.append("FMASK")
                if seg_opera:
                    sel_sources.append("BWTR")
                sum_tag = "_".join(sel_sources) if sel_sources else "DATA"

                def _name(kind: str, tag: str, ext: str) -> str:
                    # kind: 'Water' or 'Land'
                    # tag:  e.g. 'FMASK', 'BWTR', 'REFL', 'PIXEL', or combined sum tag like 'FMASK_BWTR'
                    return f"{kind}_{tag}_{seg_token}.{ext}"

                # Route outputs for ALL selected sources + optional SUM (mirrors non-segmented behavior).
                # Note: whether individual sources are actually produced depends on the user's selections
                # (USE_REFL/USE_QA_PIXEL/USE_FMASK/USE_BWTR) and the chosen mode; we only set destinations
                # for the outputs that correspond to selected inputs.
                # IMPORTANT: Use per-segment availability (seg_refl/seg_qa/seg_sent/seg_opera) to avoid
                # creating empty outputs for sources that are not present in this segment.
                if seg_refl and mode in ("refl", "both"):
                    _set_seg_dest(self.OUT_REFL_TIF, _p(_name("Water", "REFL", "tif")))
                    _set_seg_dest(self.OUT_REFL_LAND_TIF, _p(_name("Land", "REFL", "tif")))
                    _set_seg_dest(self.OUT_REFL_VEC, _p(_name("Water", "REFL", "shp")))
                    _set_seg_dest(self.OUT_REFL_LAND_VEC, _p(_name("Land", "REFL", "shp")))

                if seg_qa and mode in ("pixel", "both"):
                    _set_seg_dest(self.OUT_PIXEL_TIF, _p(_name("Water", "PIXEL", "tif")))
                    _set_seg_dest(self.OUT_PIXEL_LAND_TIF, _p(_name("Land", "PIXEL", "tif")))
                    _set_seg_dest(self.OUT_PIXEL_VEC, _p(_name("Water", "PIXEL", "shp")))
                    _set_seg_dest(self.OUT_PIXEL_LAND_VEC, _p(_name("Land", "PIXEL", "shp")))

                if seg_sent and mode in ("pixel", "both"):
                    _set_seg_dest(self.OUT_FMASK_TIF, _p(_name("Water", "FMASK", "tif")))
                    _set_seg_dest(self.OUT_FMASK_LAND_TIF, _p(_name("Land", "FMASK", "tif")))
                    _set_seg_dest(self.OUT_FMASK_VEC, _p(_name("Water", "FMASK", "shp")))
                    _set_seg_dest(self.OUT_FMASK_LAND_VEC, _p(_name("Land", "FMASK", "shp")))

                if seg_opera and mode in ("pixel", "both"):
                    _set_seg_dest(self.OUT_BWTR_TIF, _p(_name("Water", "BWTR", "tif")))
                    _set_seg_dest(self.OUT_BWTR_LAND_TIF, _p(_name("Land", "BWTR", "tif")))
                    _set_seg_dest(self.OUT_BWTR_VEC, _p(_name("Water", "BWTR", "shp")))
                    _set_seg_dest(self.OUT_BWTR_LAND_VEC, _p(_name("Land", "BWTR", "shp")))

                # SUM outputs (if requested). In BOTH mode this corresponds to the combined classification.
                if bool(p2.get(self.DO_SUM)):
                    _set_seg_dest(self.OUT_SUM_TIF, _p(_name("Water", sum_tag, "tif")))
                    _set_seg_dest(self.OUT_SUM_LAND_TIF, _p(_name("Land", sum_tag, "tif")))
                    _set_seg_dest(self.OUT_SUM_VEC, _p(_name("Water", sum_tag, "shp")))
                    _set_seg_dest(self.OUT_SUM_LAND_VEC, _p(_name("Land", sum_tag, "shp")))

                # Also route the log output (if enabled) into the segment folder.
                try:
                    if p2.get(self.OUT_LOG) not in (None, "", "TEMPORARY_OUTPUT"):
                        # If user explicitly chose a log file, keep it.
                        pass
                    else:
                        p2[self.OUT_LOG] = ""
                        p2["_SWM_SEG_LOG_PATH"] = _p(f"Log_{sum_tag}_{seg_token}.txt")
                except Exception:
                    pass

                # Scale per-segment core progress (0-100) into overall progress based on
                # completed *files* out of total files.
                sub_feedback = _ScaledFeedback(feedback, done_files, len(sub_layers), total_files)
                sub_feedback.pushInfo(f"[SEG] Running segment {y_label} + {m_label} ({len(sub_layers)} layers)")

                # HARD RULE: when segmenting-by-date, never write any raster/vector outputs into the
                # QGIS project / temporary layer store. We enforce this by using a fresh temporary
                # layer store for each segment (discarded after the segment completes), so even if
                # a child algorithm produces a temporary layer it cannot leak into the project's
                # processing context.
                _orig_tmp_store = None
                try:
                    from qgis.core import QgsMapLayerStore
                    _orig_tmp_store = context.temporaryLayerStore()
                    context.setTemporaryLayerStore(QgsMapLayerStore())
                except Exception:
                    _orig_tmp_store = None

                seg_results = self._process_core(p2, context, sub_feedback, sub_layers)

                try:
                    if _orig_tmp_store is not None:
                        context.setTemporaryLayerStore(_orig_tmp_store)
                except Exception:
                    pass

                if first_primary_results is None:
                    first_primary_results = seg_results

                # Collect written paths from the "LIST" outputs if available
                def _extend_from_key(key, acc):
                    v = seg_results.get(key) or ""
                    if isinstance(v, str) and v.strip():
                        acc.extend([line.strip() for line in v.splitlines() if line.strip()])

                _extend_from_key(self.OUT_REFL_TIF_LIST, all_refl_tif)
                _extend_from_key(self.OUT_REFL_LAND_TIF_LIST, all_refl_land_tif)
                _extend_from_key(self.OUT_PIXEL_TIF_LIST, all_pix_tif)
                _extend_from_key(self.OUT_PIXEL_LAND_TIF_LIST, all_pix_land_tif)
                _extend_from_key(self.OUT_SUM_TIF_LIST, all_sum_tif)
                _extend_from_key(self.OUT_SUM_LAND_TIF_LIST, all_sum_land_tif)

                # Segment log
                seg_log = seg_results.get(self.OUT_LOG) or ""
                if isinstance(seg_log, str) and seg_log.strip():
                    log_lines.append(f"--- Segment {y_label} + {m_label} ---")
                    log_lines.append(seg_log.strip())
                    log_lines.append("")

                # Update segmentation progress based on completed *files*.
                done_files += len(sub_layers)
                try:
                    if total_files > 0:
                        feedback.setProgress(int(100 * done_files / total_files))
                        if hasattr(feedback, "setProgressText"):
                            feedback.setProgressText(f"[SEG] {done_files}/{total_files} files complete ({i + 1}/{n} segments)")
                except Exception:
                    pass

            # Final results: expose aggregated file lists and a combined log.
            # IMPORTANT: For segmented runs we do NOT return raster/vector destinations,
            # so QGIS will not attempt to auto-load gigabytes of outputs into the project.
            results = {}

            if QgsProcessingOutputString is not None:
                results[self.OUT_REFL_TIF_LIST] = "\n".join(all_refl_tif)
                results[self.OUT_REFL_LAND_TIF_LIST] = "\n".join(all_refl_land_tif)
                results[self.OUT_PIXEL_TIF_LIST] = "\n".join(all_pix_tif)
                results[self.OUT_PIXEL_LAND_TIF_LIST] = "\n".join(all_pix_land_tif)
                results[self.OUT_SUM_TIF_LIST] = "\n".join(all_sum_tif)
                results[self.OUT_SUM_LAND_TIF_LIST] = "\n".join(all_sum_land_tif)
                results[self.OUT_LOG] = "\n".join(log_lines).strip()

            feedback.setProgress(100)
            return results

    def _process_core(self, parameters, context, feedback, layers):
            refl_tifs, qa_pixel_tifs, sentinel_tifs, opera_tifs = self._categorize(layers)

            # Internal flag: during segmentation runs we should skip segments that have no compatible
            # inputs for the requested mode(s) instead of failing the whole job.
            seg_run = bool(parameters.get("_SWM_SEGMENTED", False))

            # Respect guided-UI source selections (run ONLY what user selected)
            use_refl = parameters.get('USE_REFL', True)
            use_qa = parameters.get('USE_QA_PIXEL', True)
            use_fmask = parameters.get('USE_FMASK', True)
            use_bwtr = parameters.get('USE_BWTR', True)
            if not use_refl:
                refl_tifs = []
            if not use_qa:
                qa_pixel_tifs = []
            if not use_fmask:
                sentinel_tifs = []
            if not use_bwtr:
                opera_tifs = []



            # Re-resolve UI parameters here because _process_core may be called multiple times (segmentation).
            mode_idx = self.parameterAsEnum(parameters, self.MODE, context)
            mode = ["refl", "pixel", "both"][int(mode_idx) if mode_idx is not None else 0]

            vec_idx = self.parameterAsEnum(parameters, self.VEC_WRITE, context)
            vec_mode = ["water", "land", "both"][int(vec_idx) if vec_idx is not None else 0]
            write_water_vec = vec_mode in ("water", "both")
            write_land_vec = vec_mode in ("land", "both")

            write_water_tiffs = self.parameterAsBool(parameters, self.WRITE_TIFFS, context)
            write_land_tiffs = self.parameterAsBool(parameters, self.WRITE_LAND_TIFFS, context)
            write_any_tiffs = write_water_tiffs or write_land_tiffs

            use_sentinel = self.parameterAsBool(parameters, self.USE_SENTINEL, context)
            sentinel_in_sum = self.parameterAsBool(parameters, self.SENTINEL_SUM_WITH_LANDSAT, context)

            use_opera = self.parameterAsBool(parameters, self.USE_OPERA, context)
            opera_in_sum = self.parameterAsBool(parameters, self.OPERA_SUM_WITH_LANDSAT, context)

            # Stacks for "pixel-like" inputs. IMPORTANT: QA_PIXEL, FMASK, and BWTR are kept separate so:
            #   - QA_PIXEL outputs reflect QA_PIXEL only
            #   - FMASK outputs reflect FMASK only
            #   - BWTR outputs reflect BWTR only
            # These sources can still be optionally included in SUM (BOTH mode) via the *_SUM_WITH_LANDSAT flags.
            qapixel_tifs = list(qa_pixel_tifs)
            fmask_tifs = list(sentinel_tifs) if use_sentinel else []
            bwtr_tifs = list(opera_tifs) if use_opera else []

            # Sanity checks
            pixel_like_tifs = (qapixel_tifs or fmask_tifs or bwtr_tifs)

            # In REFL mode, REFL inputs are required
            if mode == "refl" and not refl_tifs:
                msg = "Mode includes REFL, but no *_refl raster layers were found for this date segment; skipping segment."
                if seg_run:
                    try:
                        feedback.pushInfo("[SEG][SKIP] " + msg)
                    except Exception:
                        pass
                    return {
                        self.OUT_LOG: "[SEG][SKIP] " + msg,
                        self.OUT_REFL_TIF_LIST: "",
                        self.OUT_REFL_LAND_TIF_LIST: "",
                        self.OUT_PIXEL_TIF_LIST: "",
                        self.OUT_PIXEL_LAND_TIF_LIST: "",
                        self.OUT_SUM_TIF_LIST: "",
                        self.OUT_SUM_LAND_TIF_LIST: "",
                    }
                raise QgsProcessingException(
                    "Mode includes REFL, but no *_refl raster layers were found in the selected inputs."
                )

            # In PIXEL mode, at least one pixel-like input is required
            if mode == "pixel" and not pixel_like_tifs:
                msg = ("Mode includes PIXEL, but no compatible PIXEL rasters were found for this date segment; skipping segment. "
                       "PIXEL mode requires Landsat 'QA_PIXEL' and/or Sentinel-2 HLS S30 'Fmask' layers and/or OPERA DSWx-HLS 'BWTR' layers.")
                if seg_run:
                    try:
                        feedback.pushInfo("[SEG][SKIP] " + msg)
                    except Exception:
                        pass
                    return {
                        self.OUT_LOG: "[SEG][SKIP] " + msg,
                        self.OUT_REFL_TIF_LIST: "",
                        self.OUT_REFL_LAND_TIF_LIST: "",
                        self.OUT_PIXEL_TIF_LIST: "",
                        self.OUT_PIXEL_LAND_TIF_LIST: "",
                        self.OUT_SUM_TIF_LIST: "",
                        self.OUT_SUM_LAND_TIF_LIST: "",
                    }
                raise QgsProcessingException(
                    "Mode includes PIXEL, but no compatible PIXEL rasters were found in the selected inputs. "
                    "PIXEL mode requires Landsat 'QA_PIXEL' and/or Sentinel-2 HLS S30 'Fmask' layers and/or OPERA DSWx-HLS 'BWTR' layers."
                )

            # In BOTH mode, run whatever is available; only error if nothing at all was provided
            if mode == "both" and (not refl_tifs) and (not pixel_like_tifs):
                raise QgsProcessingException(
                    "No compatible input rasters were found in the selected inputs. "
                    "Provide at least one of: Landsat *_refl, Landsat QA_PIXEL, Sentinel-2 HLS S30 Fmask, or OPERA DSWx-HLS BWTR."
                )


            # REFL-only RGB options (hidden in Advanced section in the UI)
            bg_rgb = _parse_rgb_triplet(self.parameterAsString(parameters, self.BG_RGB, context))
            keep_default = self.parameterAsBool(parameters, self.KEEP_DEFAULT, context)

            # PIXEL water code customization (Advanced)
            pixel_keep_default = self.parameterAsBool(parameters, self.PIXEL_KEEP_DEFAULT_WATER, context)
            if pixel_keep_default:
                pixel_water_vals_457 = [5504]
                pixel_water_vals_89 = [21952]
            else:
                pixel_water_vals_457 = _parse_int_list(self.parameterAsString(parameters, self.PIXEL_WATER_VALUES_457, context), [5504])
                pixel_water_vals_89 = _parse_int_list(self.parameterAsString(parameters, self.PIXEL_WATER_VALUES_89, context), [21952])

            # Sentinel-2 HLS Fmask water category (Advanced)
            s_mode_idx = self.parameterAsEnum(parameters, self.SENTINEL_WATER_MODE, context)
            s_mode = ["all", "pure", "custom"][int(s_mode_idx) if s_mode_idx is not None else 0]
            s_custom_vals = _parse_int_list(
                self.parameterAsString(parameters, self.SENTINEL_WATER_VALUES_CUSTOM, context),
                [32, 96, 160, 224],
            )

            if keep_default:
                water_ranges = ((0, 49), (0, 49), (11, 255))
            else:
                # Thresholds come either from a matrix (newer QGIS) or a compact string (QGIS <= 3.10).
                vals = []
                m = None
                if hasattr(self, "parameterAsMatrix"):
                    try:
                        m = self.parameterAsMatrix(parameters, self.RGB_THRESHOLDS, context)
                    except Exception:
                        m = None

                if isinstance(m, (list, tuple)):
                    if len(m) == 1 and isinstance(m[0], (list, tuple)):
                        vals = list(m[0])
                    else:
                        vals = list(m)
                else:
                    s = self.parameterAsString(parameters, self.RGB_THRESHOLDS, context)
                    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
                    for p in parts:
                        vals.append(p)

                while len(vals) < 6:
                    vals.append(0)

                try:
                    rmin, rmax, gmin, gmax, bmin, bmax = [int(float(v)) for v in vals[:6]]
                except Exception:
                    rmin, rmax, gmin, gmax, bmin, bmax = (0, 49, 0, 49, 11, 255)

                if rmin > rmax:
                    rmin, rmax = rmax, rmin
                if gmin > gmax:
                    gmin, gmax = gmax, gmin
                if bmin > bmax:
                    bmin, bmax = bmax, bmin

                water_ranges = ((rmin, rmax), (gmin, gmax), (bmin, bmax))

            smoothify = self.parameterAsBool(parameters, self.SMOOTHIFY, context)
            smoothify_iters = int(self.parameterAsInt(parameters, self.SMOOTHIFY_ITERS, context))
            smoothify_weight = float(self.parameterAsDouble(parameters, self.SMOOTHIFY_WEIGHT, context))
            smoothify_presimplify_m = float(self.parameterAsDouble(parameters, self.SMOOTHIFY_PRESIMPLIFY_M, context))

            pixel_smooth = self.parameterAsBool(parameters, self.PIXEL_SMOOTH, context)
            pixel_smooth_size = int(self.parameterAsInt(parameters, self.PIXEL_SMOOTH_SIZE, context))
            if pixel_smooth_size < 1:
                pixel_smooth_size = 1
            if pixel_smooth_size % 2 == 0:
                pixel_smooth_size += 1

            if smoothify and pixel_smooth:
                raise QgsProcessingException(
                    "Choose only one smoothing option: either Pixel smoothing OR Smoothify."
                )

            do_sum = self.parameterAsBool(parameters, self.DO_SUM, context)
            # Legacy switch: older guided UIs used a dedicated FMASK+BWTR SUM checkbox.
            # Newer UIs use DO_SUM to indicate "sum all enabled sources".
            do_fmask_bwtr_sum = self.parameterAsBool(parameters, self.DO_FMASK_BWTR_SUM, context)
            do_pixel_sum = bool(do_sum or do_fmask_bwtr_sum)
            refl_resample_alg = "cubic"  # REFL is continuous: use cubic when warping/resampling

            results = {}
            if QgsProcessingOutputString is not None:
                results[self.OUT_LOG] = ""

            # Internal flag: during segmentation runs we want exact filenames in the chosen folder,
            # and we do NOT want to auto-load outputs into QGIS.
            seg_run = bool(parameters.get("_SWM_SEGMENTED", False))

            # Resolve output destinations.
            # IMPORTANT: In segmented runs, avoid parameterAsOutputLayer() because it registers
            # outputs for loading into QGIS on completion (which can crash QGIS for many segments).
            if seg_run:
                def _get_out(key: str):
                    v = parameters.get(f'_SWM_SEG_DEST_{key}', None)
                    if v in (None, ''):
                        v = parameters.get(key)
                    return str(v) if v not in (None, "") else None

                refl_out_vec = _get_out(self.OUT_REFL_VEC)
                refl_out_land_vec = _get_out(self.OUT_REFL_LAND_VEC)
                pixel_out_vec = _get_out(self.OUT_PIXEL_VEC)
                pixel_out_land_vec = _get_out(self.OUT_PIXEL_LAND_VEC)

                # Standalone FMASK/BWTR vectors in segmented runs
                fmask_out_vec = _get_out(self.OUT_FMASK_VEC)
                fmask_out_land_vec = _get_out(self.OUT_FMASK_LAND_VEC)
                bwtr_out_vec = _get_out(self.OUT_BWTR_VEC)
                bwtr_out_land_vec = _get_out(self.OUT_BWTR_LAND_VEC)

                sum_out_vec = _get_out(self.OUT_SUM_VEC)
                sum_out_land_vec = _get_out(self.OUT_SUM_LAND_VEC)

                refl_out_water_tif_base = _get_out(self.OUT_REFL_TIF) if write_water_tiffs else None
                refl_out_land_tif_base = _get_out(self.OUT_REFL_LAND_TIF) if write_land_tiffs else None
                pixel_out_water_tif_base = _get_out(self.OUT_PIXEL_TIF) if write_water_tiffs else None
                pixel_out_land_tif_base = _get_out(self.OUT_PIXEL_LAND_TIF) if write_land_tiffs else None
                fmask_out_water_tif_base = _get_out(self.OUT_FMASK_TIF) if write_water_tiffs else None
                fmask_out_land_tif_base = _get_out(self.OUT_FMASK_LAND_TIF) if write_land_tiffs else None
                bwtr_out_water_tif_base = _get_out(self.OUT_BWTR_TIF) if write_water_tiffs else None
                bwtr_out_land_tif_base = _get_out(self.OUT_BWTR_LAND_TIF) if write_land_tiffs else None
                sum_out_water_tif_base = _get_out(self.OUT_SUM_TIF) if write_water_tiffs else None
                sum_out_land_tif_base = _get_out(self.OUT_SUM_LAND_TIF) if write_land_tiffs else None
            else:
                refl_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_REFL_VEC, context)
                refl_out_land_vec = self.parameterAsOutputLayer(parameters, self.OUT_REFL_LAND_VEC, context)
                pixel_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_PIXEL_VEC, context)
                pixel_out_land_vec = self.parameterAsOutputLayer(parameters, self.OUT_PIXEL_LAND_VEC, context)
                # In non-segmented runs, use QGIS's output resolution for vectors.
                # (The segmented-run helper _get_out is only defined in that branch.)
                fmask_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_FMASK_VEC, context)
                fmask_out_land_vec = self.parameterAsOutputLayer(parameters, self.OUT_FMASK_LAND_VEC, context)
                bwtr_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_BWTR_VEC, context)
                bwtr_out_land_vec = self.parameterAsOutputLayer(parameters, self.OUT_BWTR_LAND_VEC, context)

                # Resolve raster destinations early so they can be used as directory anchors when
                # FMASK/BWTR vector outputs are left as TEMPORARY_OUTPUT and the user has not set
                # explicit polygon output paths.
                refl_out_water_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_REFL_TIF, context) if write_water_tiffs else None
                refl_out_land_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_REFL_LAND_TIF, context) if write_land_tiffs else None
                pixel_out_water_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_PIXEL_TIF, context) if write_water_tiffs else None
                pixel_out_land_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_PIXEL_LAND_TIF, context) if write_land_tiffs else None
                fmask_out_water_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_FMASK_TIF, context) if write_water_tiffs else None
                fmask_out_land_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_FMASK_LAND_TIF, context) if write_land_tiffs else None
                bwtr_out_water_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_BWTR_TIF, context) if write_water_tiffs else None
                bwtr_out_land_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_BWTR_LAND_TIF, context) if write_land_tiffs else None


                # SUM outputs (only used when run_both is enabled, but must always be defined for downstream anchoring)
                sum_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_SUM_VEC, context)
                sum_out_land_vec = self.parameterAsOutputLayer(parameters, self.OUT_SUM_LAND_VEC, context)
                sum_out_water_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_SUM_TIF, context) if write_water_tiffs else None
                sum_out_land_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_SUM_LAND_TIF, context) if write_land_tiffs else None




                # If FMASK/BWTR vector outputs are left as TEMPORARY_OUTPUT (advanced params not set),
                # write them alongside the primary polygon outputs so users always get the files they expect.
                fmask_vec_raw = fmask_out_vec
                fmask_land_vec_raw = fmask_out_land_vec
                bwtr_vec_raw = bwtr_out_vec
                bwtr_land_vec_raw = bwtr_out_land_vec

                def _neighbor_vec_path(base_vec: str | None, filename: str) -> str | None:
                    # Only build neighbor paths from real, user-writable destinations.
                    if (base_vec is None) or (str(base_vec).strip() in ('', 'TEMPORARY_OUTPUT', 'temporary_output')):
                        return None
                    if not base_vec:
                        return None
                    try:
                        p = Path(base_vec)
                        return str(p.parent / filename)
                    except Exception:
                        return None

                def _is_temp_out(raw) -> bool:
                    if raw is None:
                        return True
                    s = str(raw).strip()
                    return (s == '') or (s == 'TEMPORARY_OUTPUT') or (s.lower() == 'temporary_output')

                def _good_base(v) -> bool:
                    return (v is not None) and (str(v).strip() not in ('', 'TEMPORARY_OUTPUT', 'temporary_output'))

                # Prefer an explicit QA_PIXEL/REFL vector output path as the anchor directory.
                base_vec_dir = None
                for _cand in (pixel_out_vec, refl_out_vec, sum_out_vec, pixel_out_land_vec, refl_out_land_vec, sum_out_land_vec):
                    if _good_base(_cand):
                        base_vec_dir = _cand
                        break
                if not base_vec_dir:
                    base_vec_dir = pixel_out_vec or refl_out_vec or pixel_out_land_vec or refl_out_land_vec

                # If the anchor is still TEMP/empty (common when polygon outputs are left unset),
                # fall back to the directory of any explicit raster output destination (these are typically set).
                if not _good_base(base_vec_dir):
                    for _rcand in (
                        pixel_out_water_tif_base, refl_out_water_tif_base,
                        pixel_out_land_tif_base, refl_out_land_tif_base,
                        fmask_out_water_tif_base, bwtr_out_water_tif_base,
                        fmask_out_land_tif_base, bwtr_out_land_tif_base,
                    ):
                        if _good_base(_rcand):
                            base_vec_dir = _rcand
                            break

                # If FMASK/BWTR vector outputs are unset or temporary, write them next to the anchor outputs.
                # In segmented-by-date runs we set explicit per-segment output paths; don't emit
                # extra "Water/Land Mask FMASK/BWTR" neighbor files.
                if (not parameters.get('SEGMENT_BY_DATE', False)) and (not parameters.get('_SWM_SEGMENTED', False)):
                    if _is_temp_out(fmask_vec_raw) or not _good_base(fmask_out_vec):
                        fmask_out_vec = _neighbor_vec_path(base_vec_dir, 'Water Mask FMASK.shp') or fmask_out_vec
                    if _is_temp_out(fmask_land_vec_raw) or not _good_base(fmask_out_land_vec):
                        fmask_out_land_vec = _neighbor_vec_path(base_vec_dir, 'Land Mask FMASK.shp') or fmask_out_land_vec
                    if _is_temp_out(bwtr_vec_raw) or not _good_base(bwtr_out_vec):
                        bwtr_out_vec = _neighbor_vec_path(base_vec_dir, 'Water Mask BWTR.shp') or bwtr_out_vec
                    if _is_temp_out(bwtr_land_vec_raw) or not _good_base(bwtr_out_land_vec):
                        bwtr_out_land_vec = _neighbor_vec_path(base_vec_dir, 'Land Mask BWTR.shp') or bwtr_out_land_vec
                sum_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_SUM_VEC, context)
                sum_out_land_vec = self.parameterAsOutputLayer(parameters, self.OUT_SUM_LAND_VEC, context)
                sum_out_water_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_SUM_TIF, context) if write_water_tiffs else None
                sum_out_land_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_SUM_LAND_TIF, context) if write_land_tiffs else None


            # Segmented runs should write the same set of outputs as non-segmented runs
            # (per-segment folder routing is handled upstream via _SWM_SEG_DEST_* overrides).
            TIFF_FOLDER = "Classification Tiffs"

            VEC_FOLDER = "Classification Polygons"

            def _ensure_subfolder(parent: Path, folder_name: str) -> Path:
                """Return a writable subfolder path (creates if needed)."""
                try:
                    if parent.name.lower() == folder_name.lower():
                        out_dir = parent
                    else:
                        out_dir = parent / folder_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    return out_dir
                except Exception:
                    # If the folder cannot be created (permissions, etc.), fall back to parent.
                    return parent

            def _tif_base_in_folder(base_path: str) -> str:
                """Place TIFF outputs under the 'Classification Tiffs' folder, keeping the filename as-is."""
                base = Path(base_path)
                if base.suffix.lower() not in {".tif", ".tiff"}:
                    base = base.with_suffix(".tif")
                out_dir = _ensure_subfolder(base.parent, TIFF_FOLDER)
                return str(out_dir / base.name)

            def _group_raster_path(base_path: str, pr: tuple[str, str]) -> Path:
                """Append _PPPRRR token to the base raster name, writing under 'Classification Tiffs'."""
                base = Path(_tif_base_in_folder(base_path))
                token = f"{pr[0]}{pr[1]}"
                return base.with_name(f"{base.stem}_{token}{base.suffix}")

            def _vector_path_in_folder(base_path: str, filename_no_ext: str) -> str:
                """Place polygon outputs under the 'Classification Polygons' folder and use clear names."""
                base = Path(base_path)
                out_dir = _ensure_subfolder(base.parent, VEC_FOLDER)
                # Always write shapefiles for polygon outputs
                return str(out_dir / f"{filename_no_ext}.shp")

            # Normalize output destinations into subfolders + friendly names
            if not seg_run:
                try:
                    if _good_base(refl_out_vec):
                        refl_out_vec = _vector_path_in_folder(refl_out_vec, "Water Mask REFL")
                    if _good_base(refl_out_land_vec):
                        refl_out_land_vec = _vector_path_in_folder(refl_out_land_vec, "Land Mask REFL")
                    if _good_base(pixel_out_vec):
                        pixel_out_vec = _vector_path_in_folder(pixel_out_vec, "Water Mask QA_PIXEL")
                    if _good_base(pixel_out_land_vec):
                        pixel_out_land_vec = _vector_path_in_folder(pixel_out_land_vec, "Land Mask QA_PIXEL")
                    if _good_base(fmask_out_vec):
                        fmask_out_vec = _vector_path_in_folder(fmask_out_vec, "Water Mask FMASK")
                    if _good_base(fmask_out_land_vec):
                        fmask_out_land_vec = _vector_path_in_folder(fmask_out_land_vec, "Land Mask FMASK")
                    if _good_base(bwtr_out_vec):
                        bwtr_out_vec = _vector_path_in_folder(bwtr_out_vec, "Water Mask BWTR")
                    if _good_base(bwtr_out_land_vec):
                        bwtr_out_land_vec = _vector_path_in_folder(bwtr_out_land_vec, "Land Mask BWTR")
                    if _good_base(sum_out_vec):
                        sum_out_vec = _vector_path_in_folder(sum_out_vec, "Water Mask SUM")
                    if _good_base(sum_out_land_vec):
                        sum_out_land_vec = _vector_path_in_folder(sum_out_land_vec, "Land Mask SUM")

                except Exception:
                    pass
    
                # Place TIFF outputs under 'Classification Tiffs' (filenames stay the same)
                if refl_out_water_tif_base:
                    refl_out_water_tif_base = _tif_base_in_folder(refl_out_water_tif_base)
                if refl_out_land_tif_base:
                    refl_out_land_tif_base = _tif_base_in_folder(refl_out_land_tif_base)
                if pixel_out_water_tif_base:
                    pixel_out_water_tif_base = _tif_base_in_folder(pixel_out_water_tif_base)
                if pixel_out_land_tif_base:
                    pixel_out_land_tif_base = _tif_base_in_folder(pixel_out_land_tif_base)
                if fmask_out_water_tif_base:
                    fmask_out_water_tif_base = _tif_base_in_folder(fmask_out_water_tif_base)
                if fmask_out_land_tif_base:
                    fmask_out_land_tif_base = _tif_base_in_folder(fmask_out_land_tif_base)
                if bwtr_out_water_tif_base:
                    bwtr_out_water_tif_base = _tif_base_in_folder(bwtr_out_water_tif_base)
                if bwtr_out_land_tif_base:
                    bwtr_out_land_tif_base = _tif_base_in_folder(bwtr_out_land_tif_base)
                if sum_out_water_tif_base:
                    sum_out_water_tif_base = _tif_base_in_folder(sum_out_water_tif_base)
                if sum_out_land_tif_base:
                    sum_out_land_tif_base = _tif_base_in_folder(sum_out_land_tif_base)
    
    
            # Group inputs by WRS Path/Row
            refl_groups = _group_paths_by_pathrow(refl_tifs) if refl_tifs else {}
            qapixel_groups = _group_paths_by_pathrow(qapixel_tifs) if qapixel_tifs else {}

            log_lines = []
            log_lines.append(f"Mode: {mode.upper()}")
            log_lines.append(f"Shapefiles: {vec_mode.upper()}")
            log_lines.append(f"Write Water Classification Count TIFFs: {'YES' if write_water_tiffs else 'NO'}")
            log_lines.append(f"Write Land Classification Count TIFFs: {'YES' if write_land_tiffs else 'NO'}")
            log_lines.append(f"REFL layers: {len(refl_tifs)} (groups: {len(refl_groups)})")
            log_lines.append(f"QA_PIXEL layers: {len(qapixel_tifs)} (groups: {len(qapixel_groups)})")

            if refl_groups:
                grp = ", ".join([f"{k[0]}/{k[1]}" for k in refl_groups.keys()])
                log_lines.append(f"REFL Path/Row groups: {grp}")
            if qapixel_groups:
                grp = ", ".join([f"{k[0]}/{k[1]}" for k in qapixel_groups.keys()])
                log_lines.append(f"QA_PIXEL Path/Row groups: {grp}")

            # Expanded settings summary (also shown live in the progress window)
            if mode in ("refl", "both"):
                rmin, rmax = water_ranges[0]
                gmin, gmax = water_ranges[1]
                bmin, bmax = water_ranges[2]
                log_lines.append(
                    f"REFL thresholds: R[{rmin},{rmax}] G[{gmin},{gmax}] B[{bmin},{bmax}]  (bg_rgb={bg_rgb})"
                )
            log_lines.append(f"Pixel smoothing: {'YES' if pixel_smooth else 'NO'} (kernel={pixel_smooth_size}px; applied before polygonize for vectors)")
            log_lines.append(f"Smoothify: {'YES' if smoothify else 'NO'} (iters={smoothify_iters}, weight={smoothify_weight}, pre_simplify_m={smoothify_presimplify_m:g}; applied once after final merge)")
            if mode == "both":
                log_lines.append(f"Write SUM outputs: {'YES' if do_sum else 'NO'}")

            # Stream the summary to the UI log so users immediately see what is happening.
            feedback.pushInfo("— Landsat Water Mask —")
            for ln in log_lines:
                feedback.pushInfo(ln)

            try:
                feedback.setProgress(2)
                if hasattr(feedback, "setProgressText"):
                    feedback.setProgressText("Starting processing…")
            except Exception:
                pass

            # Per-group intermediate results (needed for BOTH/SUM)
            refl_by_pr: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, tuple, str]] = {}
            pixel_by_pr: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, tuple, str]] = {}

            refl_water_geoms: list[ogr.Geometry] = []
            refl_land_geoms: list[ogr.Geometry] = []
            pixel_water_geoms: list[ogr.Geometry] = []
            pixel_land_geoms: list[ogr.Geometry] = []
            fmask_water_geoms: list[ogr.Geometry] = []
            fmask_land_geoms: list[ogr.Geometry] = []
            bwtr_water_geoms: list[ogr.Geometry] = []
            bwtr_land_geoms: list[ogr.Geometry] = []
            sum_water_geoms: list[ogr.Geometry] = []
            sum_land_geoms: list[ogr.Geometry] = []
            any_fmask = False
            any_bwtr = False
            pixel_sizes_m: list[float] = []  # for light vector smoothing when Pixel smoothing is enabled

            refl_water_tif_written: list[Path] = []
            pixel_water_tif_written: list[Path] = []
            fmask_water_tif_written: list[Path] = []
            bwtr_water_tif_written: list[Path] = []
            sum_water_tif_written: list[Path] = []
            refl_land_tif_written: list[Path] = []
            pixel_land_tif_written: list[Path] = []
            fmask_land_tif_written: list[Path] = []
            bwtr_land_tif_written: list[Path] = []
            sum_land_tif_written: list[Path] = []

            # Helper: build water/land 0/1 masks for vectorization (optional pixel smoothing)
            # IMPORTANT: Land is NOT the inverse of water globally; it is "valid and classified as land"
            # (so NoData/background never becomes land, and BOTH-mode pipelines don't leak validity into each other).
        
            def _water_land_bins_from_counts(
                w_count: np.ndarray,
                l_count: np.ndarray,
                valid_bin: np.ndarray,
                land_from_counts: bool = False,
            ) -> tuple[np.ndarray, np.ndarray]:
                """Build 0/1 water and land masks for vectorization from count rasters.

                Notes:
                  - Water is always defined as: valid & (water_count > 0)
                  - Land has two modes:
                      * land_from_counts=False (default): land = inverse of water within valid
                        (prevents overlap between water/land for a given output)
                      * land_from_counts=True: land = valid & (land_count > 0)
                        (so multi-date outputs can represent "ever land" similarly to water)
                """
                valid = valid_bin.astype(bool)

                water_bin = (valid & (w_count > 0)).astype(np.uint8)

                if land_from_counts:
                    land_bin = (valid & (l_count > 0)).astype(np.uint8)

                    # Optional pixel-space smoothing on BOTH masks (keeps edges less jagged for large outputs)
                    if pixel_smooth and pixel_smooth_size >= 3:
                        water_bin = _pixel_majority_smooth(water_bin, valid_bin, pixel_smooth_size)
                        land_bin = _pixel_majority_smooth(land_bin, valid_bin, pixel_smooth_size)

                    water_bin = (valid & water_bin.astype(bool)).astype(np.uint8)
                    land_bin = (valid & land_bin.astype(bool)).astype(np.uint8)
                    # Prevent overlap between water and land vectors: if a pixel is water, it cannot be land.
                    land_bin = (land_bin & (~water_bin.astype(bool))).astype(np.uint8)
                    return water_bin, land_bin

                # Default: Land is inverse of water within valid pixels (prevents overlap)
                if pixel_smooth and pixel_smooth_size >= 3:
                    water_bin = _pixel_majority_smooth(water_bin, valid_bin, pixel_smooth_size)

                water_bool = valid & water_bin.astype(bool)
                land_bin = (valid & (~water_bool)).astype(np.uint8)
                water_bin = water_bool.astype(np.uint8)
                return water_bin, land_bin

            def _append_geom(dst: list, binary: np.ndarray, gt, wkt, label: str):
                g = _binary_array_to_dissolved_geom_4326(
                    binary, gt, wkt,
                    smoothify=False,
                    smoothify_iters=smoothify_iters,
                    smoothify_weight=smoothify_weight,
                )
                if g is not None:
                    dst.append(g)
                else:
                    feedback.pushInfo(f"[VEC] No {label} geometry (empty mask).")

            # --- Process by mode ---
            # IMPORTANT:
            #   - Output count rasters are computed per *unique acquisition date* (YYYYMMDD) and are NOT split by Path/Row.
            #   - The output raster grid covers the UNION extent of all processed rasters (matching the shapefile extent).

            if mode == "refl":
                feedback.pushInfo(f"[REFL] Processing {len(refl_tifs)} REFL raster(s) (merged across all Path/Row groups; per-date counting).")
                stack_fb = _ScaledFeedback(feedback, 5.0, 75.0, prefix="")

                out_water_tif = Path(refl_out_water_tif_base) if (write_water_tiffs and refl_out_water_tif_base) else None
                out_land_tif = Path(refl_out_land_tif_base) if (write_land_tiffs and refl_out_land_tif_base) else None

                w, lcnt, v, gt, wkt, w_written, l_written = _process_refl_stack(
                    tifs=refl_tifs,
                    out_water_tif=out_water_tif,
                    out_land_tif=out_land_tif,
                    out_water_vec=None,
                    out_land_vec=None,
                    bg_rgb=bg_rgb,
                    water_ranges=water_ranges,
                    refl_resample=refl_resample_alg,
                    feedback=stack_fb,
                    smoothify=False,
                    smoothify_iters=smoothify_iters,
                    smoothify_weight=smoothify_weight,
                    write_water_tiffs=bool(out_water_tif),
                    write_land_tiffs=bool(out_land_tif),
                    write_water_vec=False,
                    write_land_vec=False,
                )

                pixel_sizes_m.append(_approx_pixel_size_m_from_gt_wkt(gt, wkt))
                if w_written is not None:
                    refl_water_tif_written.append(Path(w_written))
                if l_written is not None:
                    refl_land_tif_written.append(Path(l_written))

                water_bin, land_bin = _water_land_bins_from_counts(w, lcnt, v)
                if write_water_vec:
                    _append_geom(refl_water_geoms, water_bin, gt, wkt, "REFL Water")
                if write_land_vec:
                    _append_geom(refl_land_geoms, land_bin, gt, wkt, "REFL Land")

                if feedback.isCanceled():
                    raise RuntimeError("Canceled.")

            elif mode == "pixel":
                feedback.pushInfo(
                    f"[PIXEL] Processing stacks: QA_PIXEL={len(qapixel_tifs)} FMASK={len(fmask_tifs)} BWTR={len(bwtr_tifs)} "
                    f"(each merged across all Path/Row groups; per-date counting)."
                )

                def _run_pixel_like_stack(label, tifs, out_water_base, out_land_base,
                                          out_water_vec_path, out_land_vec_path,
                                          water_geoms, land_geoms,
                                          water_tif_written_list, land_tif_written_list):
                    """Run _process_pixel_stack for one pixel-like source (QA_PIXEL / FMASK / BWTR).

                    Returns a dict with arrays + grid metadata, or None if no inputs.
                    """
                    nonlocal any_fmask, any_bwtr
                    if not tifs:
                        return None
                    # Mark presence of optional augment sources so vectors get finalized/written
                    if str(label).upper() == 'FMASK':
                        any_fmask = True
                    elif str(label).upper() == 'BWTR':
                        any_bwtr = True
                    stack_fb = _ScaledFeedback(feedback, 5.0, 75.0, prefix="")
                    out_water_tif = Path(out_water_base) if (write_water_tiffs and out_water_base) else None
                    out_land_tif = Path(out_land_base) if (write_land_tiffs and out_land_base) else None

                    w, lcnt, v, gt, wkt, w_written, l_written = _process_pixel_stack(
                        tifs=tifs,
                        pixel_water_vals_457=pixel_water_vals_457,
                        pixel_water_vals_89=pixel_water_vals_89,
                        sentinel_water_mode=s_mode,
                        sentinel_custom_values=s_custom_vals,
                        out_water_tif=out_water_tif,
                        out_land_tif=out_land_tif,
                        label_tag=f"[{label}] ",
                        feedback=stack_fb,
                        do_sum=False,
                        write_water_tiffs=bool(write_water_tiffs and out_water_tif is not None),
                        write_land_tiffs=bool(write_land_tiffs and out_land_tif is not None),
                    )

                    pixel_sizes_m.append(_approx_pixel_size_m_from_gt_wkt(gt, wkt))
                    if w_written is not None:
                        water_tif_written_list.append(Path(w_written))
                    if l_written is not None:
                        land_tif_written_list.append(Path(l_written))

                    water_bin, land_bin = _water_land_bins_from_counts(w, lcnt, v)

                    # Build geometry lists for later finalize/write stage (ensures consistent smoothing behavior)
                    if write_water_vec:
                        _append_geom(water_geoms, water_bin, gt, wkt, f"{label} Water")
                    if write_land_vec:
                        _append_geom(land_geoms, land_bin, gt, wkt, f"{label} Land")
                    return {
                        "w": w,
                        "l": lcnt,
                        "v": v,
                        "gt": gt,
                        "wkt": wkt,
                        "water_bin": water_bin,
                        "land_bin": land_bin,
                        "w_written": w_written,
                        "l_written": l_written,
                    }
                resP = _run_pixel_like_stack("QA_PIXEL", qapixel_tifs, pixel_out_water_tif_base, pixel_out_land_tif_base,
                                          pixel_out_vec, pixel_out_land_vec,
                                          pixel_water_geoms, pixel_land_geoms, pixel_water_tif_written, pixel_land_tif_written)

                resF = _run_pixel_like_stack("FMASK", fmask_tifs, fmask_out_water_tif_base, fmask_out_land_tif_base,
                                          fmask_out_vec, fmask_out_land_vec,
                                          fmask_water_geoms, fmask_land_geoms, fmask_water_tif_written, fmask_land_tif_written)

                resB = _run_pixel_like_stack("BWTR", bwtr_tifs, bwtr_out_water_tif_base, bwtr_out_land_tif_base,
                                          bwtr_out_vec, bwtr_out_land_vec,
                                          bwtr_water_geoms, bwtr_land_geoms, bwtr_water_tif_written, bwtr_land_tif_written)

                # Optional: SUM across ALL enabled PIXEL-like stacks (QA_PIXEL, FMASK, BWTR).
                # This is controlled by DO_SUM (preferred) or the legacy DO_FMASK_BWTR_SUM.
                if do_pixel_sum:
                    try:
                        stacks = [("QA_PIXEL", resP), ("FMASK", resF), ("BWTR", resB)]
                        stacks = [(lbl, rr) for (lbl, rr) in stacks if rr is not None]
                        if len(stacks) >= 2:
                            # Build a UNION target grid across ALL selected sources so SUM covers the
                            # unionized extent (not just the first stack's footprint).
                            ref_lbl, ref = stacks[0]
                            gt_ref0, wkt_ref = ref["gt"], ref["wkt"]
                            xres = abs(float(gt_ref0[1]))
                            yres = abs(float(gt_ref0[5]))

                            all_paths = []
                            for lbl, rr in stacks:
                                if lbl == "QA_PIXEL":
                                    all_paths += list(qapixel_tifs)
                                elif lbl == "FMASK":
                                    all_paths += list(fmask_tifs)
                                elif lbl == "BWTR":
                                    all_paths += list(bwtr_tifs)

                            # Fallback (shouldn't happen): if we somehow don't have paths, use the ref stack grid.
                            if all_paths:
                                gt_ref, width_ref, height_ref = _compute_union_grid(
                                    [Path(pp) for pp in all_paths],
                                    wkt_ref,
                                    xres,
                                    yres,
                                    anchor_x=float(gt_ref0[0]),
                                    anchor_y=float(gt_ref0[3]),
                                    feedback=feedback,
                                    label="[SUM] ",
                                )
                                shape_ref = (height_ref, width_ref)
                            else:
                                gt_ref = gt_ref0
                                shape_ref = ref["w"].shape

                            wS = np.zeros(shape_ref, dtype=np.uint32)
                            lS = np.zeros(shape_ref, dtype=np.uint32)

                            for lbl, rr in stacks:
                                w_i = rr["w"]; l_i = rr["l"]; gt_i = rr["gt"]; wkt_i = rr["wkt"]
                                if (w_i.shape != shape_ref) or (gt_i != gt_ref) or (wkt_i != wkt_ref):
                                    w_i = _resample_array_to_grid(w_i, gt_i, wkt_i, gt_ref, wkt_ref, shape_ref, nodata=0)
                                    l_i = _resample_array_to_grid(l_i, gt_i, wkt_i, gt_ref, wkt_ref, shape_ref, nodata=0)
                                wS += w_i.astype(np.uint32)
                                lS += l_i.astype(np.uint32)

                            wS16 = wS.astype(np.uint16)
                            lS16 = lS.astype(np.uint16)
                            vS = ((wS16 + lS16) > 0).astype(np.uint8)

                            if write_water_tiffs and sum_out_water_tif_base:
                                outp = Path(sum_out_water_tif_base)
                                _write_gtiff(outp, wS16, gt_ref, wkt_ref, nodata=0)
                                sum_water_tif_written.append(outp)
                                results[self.OUT_SUM_TIF] = str(outp)
                            if write_land_tiffs and sum_out_land_tif_base:
                                outp = Path(sum_out_land_tif_base)
                                _write_gtiff(outp, lS16, gt_ref, wkt_ref, nodata=0)
                                sum_land_tif_written.append(outp)
                                results[self.OUT_SUM_LAND_TIF] = str(outp)

                            if write_water_vec or write_land_vec:
                                # For SUM vectors, we want "ever water" AND "ever land" (non-exclusive)
                                # across all selected sources. Using land_from_counts=True prevents a pixel
                                # that was flagged as water by *any* source from being excluded from the land
                                # vector when another source/date flagged it as land.
                                waterS_bin, landS_bin = _water_land_bins_from_counts(wS16, lS16, vS, land_from_counts=True)
                                if write_water_vec:
                                    _append_geom(sum_water_geoms, waterS_bin, gt_ref, wkt_ref, "SUM Water")
                                if write_land_vec:
                                    _append_geom(sum_land_geoms, landS_bin, gt_ref, wkt_ref, "SUM Land")
                    except Exception:
                        feedback.pushInfo("[SUM] Skipped SUM in PIXEL mode (alignment/processing failed).")

                if feedback.isCanceled():
                    raise RuntimeError("Canceled.")

            elif mode == "both":
                feedback.pushInfo(
                    f"[BOTH] Processing REFL={len(refl_tifs)} + QA_PIXEL={len(qapixel_tifs)} raster(s) "
                    f"(merged across all Path/Row groups; per-date counting)."
                )
                stack_fb = _ScaledFeedback(feedback, 5.0, 75.0, prefix="")

                out_refl_water_tif = Path(refl_out_water_tif_base) if (write_water_tiffs and refl_out_water_tif_base) else None
                out_refl_land_tif = Path(refl_out_land_tif_base) if (write_land_tiffs and refl_out_land_tif_base) else None
                # Only write QA_PIXEL outputs when QA_PIXEL inputs are present.
                out_pixel_water_tif = Path(pixel_out_water_tif_base) if (qapixel_tifs and write_water_tiffs and pixel_out_water_tif_base) else None
                out_pixel_land_tif = Path(pixel_out_land_tif_base) if (qapixel_tifs and write_land_tiffs and pixel_out_land_tif_base) else None
                out_sum_water_tif = Path(sum_out_water_tif_base) if (write_water_tiffs and do_sum and sum_out_water_tif_base) else None
                out_sum_land_tif = Path(sum_out_land_tif_base) if (write_land_tiffs and do_sum and sum_out_land_tif_base) else None

                # For BOTH processing we always keep sources separate (REFL / QA_PIXEL / FMASK / BWTR).
                # If DO_SUM is enabled, SUM will combine ALL enabled sources.
                aug_selected = []
                if use_sentinel:
                    aug_selected += list(sentinel_tifs)
                if use_opera:
                    aug_selected += list(opera_tifs)

                wR, lR, wP, lP, wF, lF, wB, lB, wS, lS, gt, wkt, written = _process_both_by_acqdate(
                    tifs_refl=refl_tifs,
                    tifs_pixel=qapixel_tifs,
                    tifs_augment=aug_selected,
                    out_refl_water_tif=out_refl_water_tif,
                    out_refl_land_tif=out_refl_land_tif,
                    out_pixel_water_tif=out_pixel_water_tif,
                    out_pixel_land_tif=out_pixel_land_tif,
                    out_fmask_water_tif=fmask_out_water_tif_base,
                    out_fmask_land_tif=fmask_out_land_tif_base,
                    out_bwtr_water_tif=bwtr_out_water_tif_base,
                    out_bwtr_land_tif=bwtr_out_land_tif_base,
                    out_sum_water_tif=out_sum_water_tif,
                    out_sum_land_tif=out_sum_land_tif,
                    bg_rgb=bg_rgb,
                    water_ranges=water_ranges,
                    pixel_water_vals_457=pixel_water_vals_457,
                    pixel_water_vals_89=pixel_water_vals_89,
                    sentinel_water_mode=s_mode,
                    sentinel_custom_values=s_custom_vals,
                    refl_resample=refl_resample_alg,
                    feedback=stack_fb,
                    do_sum=do_sum,
                    write_water_tiffs=write_water_tiffs,
                    write_land_tiffs=write_land_tiffs,
                )

                # Track written rasters
                for k, lst in (
                    ("refl_water", refl_water_tif_written),
                    ("refl_land", refl_land_tif_written),
                    ("pixel_water", pixel_water_tif_written),
                    ("pixel_land", pixel_land_tif_written),
                    ("fmask_water", fmask_water_tif_written),
                    ("fmask_land", fmask_land_tif_written),
                    ("bwtr_water", bwtr_water_tif_written),
                    ("bwtr_land", bwtr_land_tif_written),
                    ("sum_water", sum_water_tif_written),
                    ("sum_land", sum_land_tif_written),
                ):
                    if written.get(k) is not None:
                        lst.append(Path(written[k]))

                # FMASK/BWTR: legacy standalone mask shapefile writer disabled (redundant).

                # Validity masks per pipeline
                vR = ((wR + lR) > 0).astype(np.uint8)
                vP = ((wP + lP) > 0).astype(np.uint8)
                vF = None
                if (wF is not None) and (lF is not None):
                    vF = ((wF + lF) > 0).astype(np.uint8)
                vB = None
                if (wB is not None) and (lB is not None):
                    vB = ((wB + lB) > 0).astype(np.uint8)
                vS = None
                if do_sum and (wS is not None) and (lS is not None):
                    vS = ((wS + lS) > 0).astype(np.uint8)

                pixel_sizes_m.append(_approx_pixel_size_m_from_gt_wkt(gt, wkt))

                waterR_bin, landR_bin = _water_land_bins_from_counts(wR, lR, vR)
                waterP_bin, landP_bin = _water_land_bins_from_counts(wP, lP, vP)
                waterF_bin = landF_bin = None
                waterB_bin = landB_bin = None
                has_fmask = (wF is not None) and (lF is not None)
                has_bwtr = (wB is not None) and (lB is not None)
                if has_fmask:
                    any_fmask = True
                if has_bwtr:
                    any_bwtr = True
                if has_fmask:
                    waterF_bin, landF_bin = _water_land_bins_from_counts(wF, lF, vF)
                if has_bwtr:
                    waterB_bin, landB_bin = _water_land_bins_from_counts(wB, lB, vB)


                if write_water_vec:
                    _append_geom(refl_water_geoms, waterR_bin, gt, wkt, "REFL Water")
                    _append_geom(pixel_water_geoms, waterP_bin, gt, wkt, "QA_PIXEL Water")
                    if has_fmask and (waterF_bin is not None):
                        _append_geom(fmask_water_geoms, waterF_bin, gt, wkt, "FMASK Water")
                    if has_bwtr and (waterB_bin is not None):
                        _append_geom(bwtr_water_geoms, waterB_bin, gt, wkt, "BWTR Water")

                if write_land_vec:
                    _append_geom(refl_land_geoms, landR_bin, gt, wkt, "REFL Land")
                    _append_geom(pixel_land_geoms, landP_bin, gt, wkt, "QA_PIXEL Land")
                    if has_fmask and (landF_bin is not None):
                        _append_geom(fmask_land_geoms, landF_bin, gt, wkt, "FMASK Land")
                    if has_bwtr and (landB_bin is not None):
                        _append_geom(bwtr_land_geoms, landB_bin, gt, wkt, "BWTR Land")

                if do_sum and (wS is not None) and (lS is not None) and (vS is not None):
                    # For SUM vectors, represent "ever water" and "ever land" as non-exclusive masks.
                    # This avoids hiding land where another source/date flagged water.
                    waterS_bin, landS_bin = _water_land_bins_from_counts(wS, lS, vS, land_from_counts=True)
                    if write_water_vec:
                        _append_geom(sum_water_geoms, waterS_bin, gt, wkt, "SUM Water")
                    if write_land_vec:
                        _append_geom(sum_land_geoms, landS_bin, gt, wkt, "SUM Land")

                if feedback.isCanceled():
                    raise RuntimeError("Canceled.")

            try:
                feedback.setProgress(75)
                if hasattr(feedback, "setProgressText"):
                    feedback.setProgressText("Preparing vector outputs…")
            except Exception:
                pass

            # SUM vectors are built directly from the SUM count rasters (pixel-level combination)
            # during per-Path/Row processing above, so no additional geometry mixing is needed here.
            # Light vector smoothing (used when Pixel smoothing is enabled and Smoothify is OFF)
            # Light vector smoothing (used when Pixel smoothing is enabled and Smoothify is OFF)
            pixel_vec_presimplify_m = 0.0
            if pixel_smooth and (not smoothify) and pixel_smooth_size >= 3:
                try:
                    rep_px_m = float(np.median(np.array(pixel_sizes_m, dtype=float))) if pixel_sizes_m else 30.0
                    # For k=3 => ~0.5 pixel; for k=5 => ~1.0 pixel, etc.
                    pixel_vec_presimplify_m = max(0.0, (float(pixel_smooth_size) - 1.0) * rep_px_m / 4.0)
                except Exception:
                    pixel_vec_presimplify_m = 15.0

            # --- Finalize merged + dissolved vectors ---
            # Vectors are dissolved per output (REFL / QA_PIXEL / SUM). SUM vectors come from the SUM count rasters,
            # so they are finalized the same way as the other outputs.
            def _finalize_and_write_vec(
                out_path: str | None,
                geoms: list[ogr.Geometry],
                label: str,
                class_id: int,
                allow_empty: bool = False,
            ) -> tuple[bool, ogr.Geometry | None]:
                # Some runs (notably segmented runs) may intentionally suppress writing certain
                # outputs by leaving the destination unset. Never crash in these cases.
                if out_path in (None, ""):
                    return False, None
                if not geoms:
                    if allow_empty:
                        try:
                            # If allow_empty but no destination was provided, just skip.
                            if out_path in (None, ""):
                                return False, None
                            _write_empty_vector_4326(Path(out_path), label=label)
                            return Path(out_path).exists(), None
                        except Exception:
                            return False, None
                    return False, None
                merged = _union_multipolygons(geoms)
                if merged is None:
                    return False, None

                # If Pixel smoothing is enabled (and Smoothify is OFF), apply a light vector smoothing here
                # to avoid jagged pixel-step edges while staying fast on large scenes.
                if pixel_smooth and (not smoothify) and pixel_vec_presimplify_m and pixel_vec_presimplify_m > 0:
                    try:
                        before_n = _ogr_total_vertex_count(merged)
                        try:
                            c = merged.Centroid()
                            lat = float(c.GetY()) if (c is not None and not c.IsEmpty()) else 0.0
                        except Exception:
                            lat = 0.0
                        tol_deg = _meters_to_degrees_tol_at_lat(float(pixel_vec_presimplify_m), lat)
                        if tol_deg > 0:
                            feedback.pushInfo(
                                f"[PIXSMOOTH] Light vector smoothing final {label} (tol≈{pixel_vec_presimplify_m:g} m ~ {tol_deg:.6g}°)…"
                            )
                            simp = merged.SimplifyPreserveTopology(tol_deg)
                            if simp is not None and not simp.IsEmpty():
                                merged = simp
                        merged = _smoothify_ogr_geometry(merged, iterations=1, weight=0.25)
                        try:
                            if hasattr(merged, "IsValid") and (not merged.IsValid()):
                                merged = merged.Buffer(0)
                        except Exception:
                            pass
                        after_n = _ogr_total_vertex_count(merged)
                        if after_n and before_n:
                            feedback.pushInfo(f"[PIXSMOOTH] Vertex count {before_n:,} → {after_n:,}")
                    except Exception:
                        feedback.pushInfo(f"[PIXSMOOTH] Vector smoothing failed for {label}; writing unsmoothed geometry.")
                # Apply Smoothify once on the final merged geometry (much faster than per-Path/Row).
                if smoothify:
                    try:
                        # Pre-simplify (topology-preserving) to massively cut vertex count before Chaikin smoothing.
                        # This keeps the same general look but avoids multi-minute smoothify runs on huge coastlines.
                        if smoothify_presimplify_m and smoothify_presimplify_m > 0:
                            before_n = _ogr_total_vertex_count(merged)
                            try:
                                c = merged.Centroid()
                                lat = float(c.GetY()) if (c is not None and not c.IsEmpty()) else 0.0
                            except Exception:
                                lat = 0.0
                            tol_deg = _meters_to_degrees_tol_at_lat(float(smoothify_presimplify_m), lat)
                            if tol_deg > 0:
                                feedback.pushInfo(
                                    f"[SMOOTH] Pre-simplify final {label} geometry (tol≈{smoothify_presimplify_m:g} m ~ {tol_deg:.6g}°)…"
                                )
                                simp = merged.SimplifyPreserveTopology(tol_deg)
                                if simp is not None and not simp.IsEmpty():
                                    merged = simp
                                after_n = _ogr_total_vertex_count(merged)
                                feedback.pushInfo(f"[SMOOTH] Vertex count {before_n:,} → {after_n:,}")

                        feedback.pushInfo(
                            f"[SMOOTH] Smoothing final {label} geometry (iters={smoothify_iters}, weight={smoothify_weight})…"
                        )
                        merged = _smoothify_ogr_geometry(merged, iterations=smoothify_iters, weight=smoothify_weight)

                        # Repair only if needed (Buffer(0) can be extremely expensive).
                        try:
                            if hasattr(merged, "IsValid") and (not merged.IsValid()):
                                feedback.pushInfo("[SMOOTH] Geometry invalid after smoothing; repairing (Buffer(0))…")
                                merged = merged.Buffer(0)
                        except Exception:
                            pass

                    except Exception:
                        feedback.pushInfo(f"[SMOOTH] Smoothing failed for {label}; writing unsmoothed geometry.")
                _write_single_feature_geom_4326(Path(out_path), merged, label=label, class_id=class_id)
                return Path(out_path).exists(), merged

# --- Write merged + dissolved vectors with progress ---
            vec_tasks = []
            if mode in ("refl", "both"):
                if write_water_vec:
                    vec_tasks.append(("REFL Water", refl_out_vec, refl_water_geoms, "Water", 1, self.OUT_REFL_VEC, False))
                if write_land_vec:
                    vec_tasks.append(("REFL Land", refl_out_land_vec, refl_land_geoms, "Land", 2, self.OUT_REFL_LAND_VEC, False))

            if mode in ("pixel", "both"):
                if write_water_vec:
                    vec_tasks.append(("QA_PIXEL Water", pixel_out_vec, pixel_water_geoms, "Water", 1, self.OUT_PIXEL_VEC, False))
                    if any_fmask:
                        vec_tasks.append(("FMASK Water", fmask_out_vec, fmask_water_geoms, "Water", 1, self.OUT_FMASK_VEC, True))
                    if any_bwtr:
                        vec_tasks.append(("BWTR Water", bwtr_out_vec, bwtr_water_geoms, "Water", 1, self.OUT_BWTR_VEC, True))
                if write_land_vec:
                    vec_tasks.append(("QA_PIXEL Land", pixel_out_land_vec, pixel_land_geoms, "Land", 2, self.OUT_PIXEL_LAND_VEC, False))
                    if any_fmask:
                        vec_tasks.append(("FMASK Land", fmask_out_land_vec, fmask_land_geoms, "Land", 2, self.OUT_FMASK_LAND_VEC, True))
                    if any_bwtr:
                        vec_tasks.append(("BWTR Land", bwtr_out_land_vec, bwtr_land_geoms, "Land", 2, self.OUT_BWTR_LAND_VEC, True))

            if do_sum and mode in ("both", "pixel"):
                if write_water_vec and sum_water_geoms:
                    vec_tasks.append(("SUM Water", sum_out_vec, sum_water_geoms, "Water", 1, self.OUT_SUM_VEC, True))
                if write_land_vec and sum_land_geoms:
                    vec_tasks.append(("SUM Land", sum_out_land_vec, sum_land_geoms, "Land", 2, self.OUT_SUM_LAND_VEC, True))


            if vec_tasks:
                # Store finalized geometries (useful for debugging/logging).
                finalized_by_key: dict[str, ogr.Geometry] = {}
                for ti, (desc, out_path, geoms, label, class_id, out_key, allow_empty) in enumerate(vec_tasks, start=1):
                    if feedback.isCanceled():
                        raise RuntimeError("Canceled.")

                    p0 = 75.0 + (ti - 1) * (20.0 / len(vec_tasks))
                    p1 = 75.0 + ti * (20.0 / len(vec_tasks))

                    try:
                        feedback.setProgress(p0)
                        if hasattr(feedback, "setProgressText"):
                            feedback.setProgressText(f"Writing {desc} vector…")
                    except Exception:
                        pass

                    wrote = False
                    merged = None

                    wrote, merged = _finalize_and_write_vec(out_path, geoms, label, class_id, allow_empty=allow_empty)

                    if wrote:
                        results[out_key] = str(out_path)
                        if merged is not None:
                            finalized_by_key[out_key] = merged

                    try:
                        feedback.setProgress(p1)
                    except Exception:
                        pass
            else:
                try:
                    feedback.setProgress(95)
                except Exception:
                    pass

    # --- Output summary (live) ---
            # Give users a concise "what was written" snapshot in the progress log.
            if mode in ("refl", "both"):
                if write_water_vec:
                    feedback.pushInfo(f"[OUTPUT] REFL Water vector: {results.get(self.OUT_REFL_VEC) or 'NOT WRITTEN'}")
                if write_land_vec:
                    feedback.pushInfo(f"[OUTPUT] REFL Land vector: {results.get(self.OUT_REFL_LAND_VEC) or 'NOT WRITTEN'}")
            if mode in ("pixel", "both"):
                if write_water_vec:
                    feedback.pushInfo(f"[OUTPUT] QA_PIXEL Water vector: {results.get(self.OUT_PIXEL_VEC) or 'NOT WRITTEN'}")
                if write_land_vec:
                    feedback.pushInfo(f"[OUTPUT] QA_PIXEL Land vector: {results.get(self.OUT_PIXEL_LAND_VEC) or 'NOT WRITTEN'}")
            if (mode in ("both", "pixel")) and do_pixel_sum:
                if write_water_vec:
                    feedback.pushInfo(f"[OUTPUT] SUM Water vector: {results.get(self.OUT_SUM_VEC) or 'NOT WRITTEN'}")
                if write_land_vec:
                    feedback.pushInfo(f"[OUTPUT] SUM Land vector: {results.get(self.OUT_SUM_LAND_VEC) or 'NOT WRITTEN'}")

            # --- TIFF outputs + logging ---
            # Only list paths when the corresponding write option is enabled.
            if write_water_tiffs:
                if refl_water_tif_written:
                    results[self.OUT_REFL_TIF] = str(refl_water_tif_written[0])
                if pixel_water_tif_written:
                    results[self.OUT_PIXEL_TIF] = str(pixel_water_tif_written[0])
                if fmask_water_tif_written:
                    results[self.OUT_FMASK_TIF] = str(fmask_water_tif_written[0])
                if bwtr_water_tif_written:
                    results[self.OUT_BWTR_TIF] = str(bwtr_water_tif_written[0])
                if sum_water_tif_written:
                    results[self.OUT_SUM_TIF] = str(sum_water_tif_written[0])

                results[self.OUT_REFL_TIF_LIST] = "\n".join(str(pp) for pp in refl_water_tif_written)
                results[self.OUT_PIXEL_TIF_LIST] = "\n".join(str(pp) for pp in pixel_water_tif_written)
                results[self.OUT_FMASK_TIF_LIST] = "\n".join(str(pp) for pp in fmask_water_tif_written)
                results[self.OUT_BWTR_TIF_LIST] = "\n".join(str(pp) for pp in bwtr_water_tif_written)
                results[self.OUT_SUM_TIF_LIST] = "\n".join(str(pp) for pp in sum_water_tif_written)
            else:
                results[self.OUT_REFL_TIF_LIST] = ""
                results[self.OUT_PIXEL_TIF_LIST] = ""
                results[self.OUT_FMASK_TIF_LIST] = ""
                results[self.OUT_BWTR_TIF_LIST] = ""
                results[self.OUT_SUM_TIF_LIST] = ""

            if write_land_tiffs:
                if refl_land_tif_written:
                    results[self.OUT_REFL_LAND_TIF] = str(refl_land_tif_written[0])
                if pixel_land_tif_written:
                    results[self.OUT_PIXEL_LAND_TIF] = str(pixel_land_tif_written[0])
                if fmask_land_tif_written:
                    results[self.OUT_FMASK_LAND_TIF] = str(fmask_land_tif_written[0])
                if bwtr_land_tif_written:
                    results[self.OUT_BWTR_LAND_TIF] = str(bwtr_land_tif_written[0])
                if sum_land_tif_written:
                    results[self.OUT_SUM_LAND_TIF] = str(sum_land_tif_written[0])

                results[self.OUT_REFL_LAND_TIF_LIST] = "\n".join(str(pp) for pp in refl_land_tif_written)
                results[self.OUT_PIXEL_LAND_TIF_LIST] = "\n".join(str(pp) for pp in pixel_land_tif_written)
                results[self.OUT_FMASK_LAND_TIF_LIST] = "\n".join(str(pp) for pp in fmask_land_tif_written)
                results[self.OUT_BWTR_LAND_TIF_LIST] = "\n".join(str(pp) for pp in bwtr_land_tif_written)
                results[self.OUT_SUM_LAND_TIF_LIST] = "\n".join(str(pp) for pp in sum_land_tif_written)
            else:
                results[self.OUT_REFL_LAND_TIF_LIST] = ""
                results[self.OUT_PIXEL_LAND_TIF_LIST] = ""
                results[self.OUT_FMASK_LAND_TIF_LIST] = ""
                results[self.OUT_BWTR_LAND_TIF_LIST] = ""
                results[self.OUT_SUM_LAND_TIF_LIST] = ""

            if write_any_tiffs:
                if mode in ("refl", "both"):
                    if write_water_tiffs:
                        feedback.pushInfo(f"[OUTPUT] REFL Water Classification Count rasters written: {len(refl_water_tif_written)}")
                    if write_land_tiffs:
                        feedback.pushInfo(f"[OUTPUT] REFL Land Classification Count rasters written: {len(refl_land_tif_written)}")
                if mode in ("pixel", "both"):
                    if write_water_tiffs:
                        feedback.pushInfo(f"[OUTPUT] QA_PIXEL Water Classification Count rasters written: {len(pixel_water_tif_written)}")
                    if write_land_tiffs:
                        feedback.pushInfo(f"[OUTPUT] QA_PIXEL Land Classification Count rasters written: {len(pixel_land_tif_written)}")
                if (mode in ("both", "pixel")) and do_pixel_sum:
                    if write_water_tiffs:
                        feedback.pushInfo(f"[OUTPUT] SUM Water Classification Count rasters written: {len(sum_water_tif_written)}")
                    if write_land_tiffs:
                        feedback.pushInfo(f"[OUTPUT] SUM Land Classification Count rasters written: {len(sum_land_tif_written)}")

            # Place individual TIFF paths at the end of the log only when requested
            if write_water_tiffs:
                if refl_water_tif_written:
                    log_lines.append("\nREFL Water Classification Count TIFF paths:")
                    log_lines.extend([f"  - {p}" for p in refl_water_tif_written])
                if pixel_water_tif_written:
                    log_lines.append("\nQA_PIXEL Water Classification Count TIFF paths:")
                    log_lines.extend([f"  - {p}" for p in pixel_water_tif_written])
                if sum_water_tif_written:
                    log_lines.append("\nSUM Water Classification Count TIFF paths:")
                    log_lines.extend([f"  - {p}" for p in sum_water_tif_written])

            if write_land_tiffs:
                if refl_land_tif_written:
                    log_lines.append("\nREFL Land Classification Count TIFF paths:")
                    log_lines.extend([f"  - {p}" for p in refl_land_tif_written])
                if pixel_land_tif_written:
                    log_lines.append("\nQA_PIXEL Land Classification Count TIFF paths:")
                    log_lines.extend([f"  - {p}" for p in pixel_land_tif_written])
                if sum_land_tif_written:
                    log_lines.append("\nSUM Land Classification Count TIFF paths:")
                    log_lines.extend([f"  - {p}" for p in sum_land_tif_written])

            if QgsProcessingOutputString is not None:
                results[self.OUT_LOG] = "\n".join(log_lines)
            feedback.setProgress(100)
            return results
