# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
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

_ACQDATE_RE = re.compile(r"_(\d{8})_")


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
    name = Path(path_or_name).name if hasattr(path_or_name, "name") else str(path_or_name)
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
    """Extract WRS-2 Path/Row from common Landsat filenames.

    Example:
      LC09_L2SP_022039_20251211_20251212_02_T1_...  -> ("022", "039")

    We look for the first "_PPPRRR_" 6-digit token in the name.
    """
    name = Path(path_or_name).name if isinstance(path_or_name, Path) else str(path_or_name)
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
    """Union/dissolve a list of polygonal geometries (assumed already in the same SRS)."""
    if not geoms:
        return None
    mp = ogr.Geometry(ogr.wkbMultiPolygon)
    for g in geoms:
        if g is None:
            continue
        gg = _ensure_multipolygon(g)
        for i in range(gg.GetGeometryCount()):
            pg = gg.GetGeometryRef(i)
            if pg is not None:
                mp.AddGeometry(pg.Clone())
    if mp.GetGeometryCount() == 0:
        return None
    try:
        return mp.UnionCascaded()
    except Exception:
        # fallback: iterative union
        out = None
        for i in range(mp.GetGeometryCount()):
            pg = mp.GetGeometryRef(i)
            if pg is None:
                continue
            out = pg.Clone() if out is None else out.Union(pg)
        return _ensure_multipolygon(out) if out is not None else None


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
    feedback,
    refl_resample: str = "near",
    smoothify: bool = False,
    smoothify_iters: int = 3,
    smoothify_weight: float = 0.2,
    write_water_tiffs: bool = False,
    write_land_tiffs: bool = False,
    write_water_vec: bool = True,
    write_land_vec: bool = False,
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
    out_water_tif: Path | None,
    out_land_tif: Path | None,
    out_water_vec: Path | None,
    out_land_vec: Path | None,
    feedback,
    pixel_water_vals_457: Sequence[int] = (5504,),
    pixel_water_vals_89: Sequence[int] = (21952,),
    smoothify: bool = False,
    smoothify_iters: int = 3,
    smoothify_weight: float = 0.2,
    write_water_tiffs: bool = False,
    write_land_tiffs: bool = False,
    write_water_vec: bool = True,
    write_land_vec: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple, str, Path | None, Path | None]:
    """Process a stack of QA_PIXEL rasters.

    Water is identified by the appropriate QA value for the Landsat mission.

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

    date_groups = _group_paths_by_acqdate(tifs)

    # Progress init
    try:
        feedback.setProgress(0)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText("[QA_PIXEL] Preparing stack…")
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
        label="[PIXEL] ",
    )

    water_count = np.zeros((height, width), dtype=np.uint16)
    land_count = np.zeros((height, width), dtype=np.uint16)
    valid_union = np.zeros((height, width), dtype=bool)

    feedback.pushInfo(f"[PIXEL] Unique acquisition dates: {len(date_groups)}")

    for i, (acq_date, paths) in enumerate(sorted(date_groups.items()), start=1):
        if feedback.isCanceled():
            raise RuntimeError("Canceled.")

        # Progress within this stack
        try:
            n_dates = max(1, len(date_groups))
            p0 = 5.0 + 85.0 * (float(i - 1) / float(n_dates))
            feedback.setProgress(p0)
            if hasattr(feedback, "setProgressText"):
                feedback.setProgressText(f"[QA_PIXEL] Processing date {i}/{n_dates}: {acq_date}")
        except Exception:
            pass

        feedback.pushInfo(f"[PIXEL]  • Date {i}/{len(date_groups)}: {acq_date} ({len(paths)} file(s))")

        date_water = np.zeros((height, width), dtype=bool)
        date_valid = np.zeros((height, width), dtype=bool)

        for path in paths:
            ds = gdal.Open(str(path))
            if ds is None:
                feedback.pushInfo(f"[PIXEL]    - Skipping (open failed): {path}")
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

            # Derive mission-specific water values from filename (04/05/07/08/09)
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
            feedback.setProgressText("[QA_PIXEL] Writing count rasters…")
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
            feedback.setProgressText("[QA_PIXEL] Done.")
    except Exception:
        pass

    return water_count, land_count, valid_bin, gt, ref_wkt, written_water_tif, written_land_tif

def _process_both_by_acqdate(
    tifs_refl: list[Path],
    tifs_pixel: list[Path],
    out_refl_water_tif: Path | None,
    out_refl_land_tif: Path | None,
    out_pixel_water_tif: Path | None,
    out_pixel_land_tif: Path | None,
    out_sum_water_tif: Path | None,
    out_sum_land_tif: Path | None,
    bg_rgb: tuple[int, int, int],
    water_ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    feedback,
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

    # Group by date for each pipeline
    refl_by_date = _group_paths_by_acqdate(tifs_refl)
    pix_by_date = _group_paths_by_acqdate(tifs_pixel)
    all_dates = sorted(set(refl_by_date.keys()) | set(pix_by_date.keys()))

    # Progress init
    try:
        feedback.setProgress(0)
        if hasattr(feedback, "setProgressText"):
            feedback.setProgressText("[BOTH] Preparing stacks…")
    except Exception:
        pass

    # Reference grid from first REFL if available, else first PIXEL
    ref_path = (tifs_refl[0] if tifs_refl else tifs_pixel[0])
    ref_ds = gdal.Open(str(ref_path))
    if ref_ds is None:
        raise RuntimeError(f"Could not open: {ref_path}")
    # Union target grid across BOTH pipelines so output rasters cover the full processed extent
    ref_gt0 = ref_ds.GetGeoTransform()
    ref_wkt = ref_ds.GetProjection()
    xres = abs(float(ref_gt0[1]))
    yres = abs(float(ref_gt0[5]))

    all_paths = list(tifs_refl) + list(tifs_pixel)
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
    sum_water_count = np.zeros((height, width), dtype=np.uint16) if do_sum else None

    refl_land_count = np.zeros((height, width), dtype=np.uint16)
    pixel_land_count = np.zeros((height, width), dtype=np.uint16)
    sum_land_count = np.zeros((height, width), dtype=np.uint16) if do_sum else None

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
        feedback.pushInfo(f"[BOTH]  • Date {i}/{len(all_dates)}: {acq_date} (REFL={len(refl_paths)} QA_PIXEL={len(pix_paths)})")

        date_refl_water = np.zeros((height, width), dtype=bool)
        date_refl_valid = np.zeros((height, width), dtype=bool)
        date_pix_water = np.zeros((height, width), dtype=bool)
        date_pix_valid = np.zeros((height, width), dtype=bool)

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

        # --- QA_PIXEL ---
        for path in pix_paths:
            ds = gdal.Open(str(path))
            if ds is None:
                feedback.pushInfo(f"[PIXEL]    - Skipping (open failed): {path}")
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

            water_val = _qa_water_value_from_filename(path.name)

            # Valid pixels: exclude explicit nodata and QA_PIXEL FILL flag (bit 0)
            valid2 = _qa_pixel_valid_mask(qa, alpha=alpha, nodata=nodata)

            is_water = valid2 & (qa == water_val)

            date_pix_water |= is_water
            date_pix_valid |= valid2
            valid_union |= valid2

        # Update counts per-date
        refl_water_count += date_refl_water.astype(np.uint16)
        refl_land_count += (date_refl_valid & (~date_refl_water)).astype(np.uint16)

        pixel_water_count += date_pix_water.astype(np.uint16)
        pixel_land_count += (date_pix_valid & (~date_pix_water)).astype(np.uint16)

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
            date_sum_valid = date_refl_valid | date_pix_valid
            date_sum_water = date_refl_water | date_pix_water
            date_sum_land = date_sum_valid & (~date_sum_water)

            sum_water_count += date_sum_water.astype(np.uint16)
            sum_land_count += date_sum_land.astype(np.uint16)

    valid_bin = valid_union.astype(np.uint8)

    written = {
        "refl_water": None,
        "refl_land": None,
        "pixel_water": None,
        "pixel_land": None,
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

    return (
        refl_water_count, refl_land_count,
        pixel_water_count, pixel_land_count,
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
    feedback,
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


class LandsatWaterMaskAlgorithm(QgsProcessingAlgorithm):
    """
    QGIS Processing algorithm.

    - Inputs: raster layers currently loaded in the project (or optionally, a selected subset)
    - Outputs: rasters/vectors returned to QGIS as Processing destinations (TEMPORARY_OUTPUT by default)
    """

    INPUT_LAYERS = "INPUT_LAYERS"
    MODE = "MODE"

    BG_RGB = "BG_RGB"
    KEEP_DEFAULT = "KEEP_DEFAULT"
    RGB_THRESHOLDS = "RGB_THRESHOLDS"

    PIXEL_KEEP_DEFAULT_WATER = "PIXEL_KEEP_DEFAULT_WATER"
    PIXEL_WATER_VALUES_457 = "PIXEL_WATER_VALUES_457"
    PIXEL_WATER_VALUES_89 = "PIXEL_WATER_VALUES_89"

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
    OUT_SUM_TIF = "OUT_SUM_TIF"
    OUT_SUM_LAND_TIF = "OUT_SUM_LAND_TIF"
    OUT_SUM_VEC = "OUT_SUM_VEC"
    OUT_SUM_LAND_VEC = "OUT_SUM_LAND_VEC"

    # Newline-separated lists of all written TIFFs (when multiple Path/Row groups exist)
    OUT_REFL_TIF_LIST = "OUT_REFL_TIF_LIST"
    OUT_REFL_LAND_TIF_LIST = "OUT_REFL_LAND_TIF_LIST"
    OUT_PIXEL_TIF_LIST = "OUT_PIXEL_TIF_LIST"
    OUT_PIXEL_LAND_TIF_LIST = "OUT_PIXEL_LAND_TIF_LIST"
    OUT_SUM_TIF_LIST = "OUT_SUM_TIF_LIST"
    OUT_SUM_LAND_TIF_LIST = "OUT_SUM_LAND_TIF_LIST"

    OUT_LOG = "OUT_LOG"

    def name(self):
        return "landsat_water_mask"

    def displayName(self):
        return "Landsat Water Mask (from loaded layers)"

    def group(self):
        return "Landsat Water Mask"

    def groupId(self):
        return "landsat_water_mask"

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
        return LandsatWaterMaskAlgorithm()

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
        refl_paths = []
        pixel_paths = []

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
                pixel_paths.append(p)

        return _dedupe_paths(refl_paths), _dedupe_paths(pixel_paths)
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

        layers = self._collect_layers(parameters, context)
        refl_tifs, pixel_tifs = self._categorize(layers)

        # Sanity checks
        if mode in ("refl", "both") and not refl_tifs:
            raise QgsProcessingException(
                "Mode includes REFL, but no *_refl raster layers were found in the selected inputs."
            )
        if mode in ("pixel", "both") and not pixel_tifs:
            raise QgsProcessingException(
                "Mode includes PIXEL, but no QA_PIXEL raster layers were found in the selected inputs."
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
        refl_resample_alg = "cubic"  # REFL is continuous: use cubic when warping/resampling

        results = {}
        if QgsProcessingOutputString is not None:
            results[self.OUT_LOG] = ""

        # Resolve output destinations
        refl_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_REFL_VEC, context)
        refl_out_land_vec = self.parameterAsOutputLayer(parameters, self.OUT_REFL_LAND_VEC, context)
        pixel_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_PIXEL_VEC, context)
        pixel_out_land_vec = self.parameterAsOutputLayer(parameters, self.OUT_PIXEL_LAND_VEC, context)
        sum_out_vec = self.parameterAsOutputLayer(parameters, self.OUT_SUM_VEC, context)
        sum_out_land_vec = self.parameterAsOutputLayer(parameters, self.OUT_SUM_LAND_VEC, context)
        refl_out_water_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_REFL_TIF, context) if write_water_tiffs else None
        refl_out_land_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_REFL_LAND_TIF, context) if write_land_tiffs else None
        pixel_out_water_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_PIXEL_TIF, context) if write_water_tiffs else None
        pixel_out_land_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_PIXEL_LAND_TIF, context) if write_land_tiffs else None
        sum_out_water_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_SUM_TIF, context) if write_water_tiffs else None
        sum_out_land_tif_base = self.parameterAsOutputLayer(parameters, self.OUT_SUM_LAND_TIF, context) if write_land_tiffs else None


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
        try:
            refl_out_vec = _vector_path_in_folder(refl_out_vec, "Water Mask REFL")
            refl_out_land_vec = _vector_path_in_folder(refl_out_land_vec, "Land Mask REFL")
            pixel_out_vec = _vector_path_in_folder(pixel_out_vec, "Water Mask PIXEL")
            pixel_out_land_vec = _vector_path_in_folder(pixel_out_land_vec, "Land Mask PIXEL")
            sum_out_vec = _vector_path_in_folder(sum_out_vec, "Water Mask SUM")
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
        if sum_out_water_tif_base:
            sum_out_water_tif_base = _tif_base_in_folder(sum_out_water_tif_base)
        if sum_out_land_tif_base:
            sum_out_land_tif_base = _tif_base_in_folder(sum_out_land_tif_base)


        # Group inputs by WRS Path/Row
        refl_groups = _group_paths_by_pathrow(refl_tifs) if refl_tifs else {}
        pixel_groups = _group_paths_by_pathrow(pixel_tifs) if pixel_tifs else {}

        log_lines = []
        log_lines.append(f"Mode: {mode.upper()}")
        log_lines.append(f"Shapefiles: {vec_mode.upper()}")
        log_lines.append(f"Write Water Classification Count TIFFs: {'YES' if write_water_tiffs else 'NO'}")
        log_lines.append(f"Write Land Classification Count TIFFs: {'YES' if write_land_tiffs else 'NO'}")
        log_lines.append(f"REFL layers: {len(refl_tifs)} (groups: {len(refl_groups)})")
        log_lines.append(f"QA_PIXEL layers: {len(pixel_tifs)} (groups: {len(pixel_groups)})")

        if refl_groups:
            grp = ", ".join([f"{k[0]}/{k[1]}" for k in refl_groups.keys()])
            log_lines.append(f"REFL Path/Row groups: {grp}")
        if pixel_groups:
            grp = ", ".join([f"{k[0]}/{k[1]}" for k in pixel_groups.keys()])
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
        sum_water_geoms: list[ogr.Geometry] = []
        sum_land_geoms: list[ogr.Geometry] = []
        pixel_sizes_m: list[float] = []  # for light vector smoothing when Pixel smoothing is enabled

        refl_water_tif_written: list[Path] = []
        pixel_water_tif_written: list[Path] = []
        sum_water_tif_written: list[Path] = []
        refl_land_tif_written: list[Path] = []
        pixel_land_tif_written: list[Path] = []
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
            feedback.pushInfo(f"[PIXEL] Processing {len(pixel_tifs)} QA_PIXEL raster(s) (merged across all Path/Row groups; per-date counting).")
            stack_fb = _ScaledFeedback(feedback, 5.0, 75.0, prefix="")

            out_water_tif = Path(pixel_out_water_tif_base) if (write_water_tiffs and pixel_out_water_tif_base) else None
            out_land_tif = Path(pixel_out_land_tif_base) if (write_land_tiffs and pixel_out_land_tif_base) else None

            w, lcnt, v, gt, wkt, w_written, l_written = _process_pixel_stack(
                tifs=pixel_tifs,
                pixel_water_vals_457=pixel_water_vals_457,
                pixel_water_vals_89=pixel_water_vals_89,
                out_water_tif=out_water_tif,
                out_land_tif=out_land_tif,
                out_water_vec=None,
                out_land_vec=None,
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
                pixel_water_tif_written.append(Path(w_written))
            if l_written is not None:
                pixel_land_tif_written.append(Path(l_written))

            water_bin, land_bin = _water_land_bins_from_counts(w, lcnt, v)
            if write_water_vec:
                _append_geom(pixel_water_geoms, water_bin, gt, wkt, "QA_PIXEL Water")
            if write_land_vec:
                _append_geom(pixel_land_geoms, land_bin, gt, wkt, "QA_PIXEL Land")

            if feedback.isCanceled():
                raise RuntimeError("Canceled.")

        elif mode == "both":
            feedback.pushInfo(
                f"[BOTH] Processing REFL={len(refl_tifs)} + QA_PIXEL={len(pixel_tifs)} raster(s) "
                f"(merged across all Path/Row groups; per-date counting)."
            )
            stack_fb = _ScaledFeedback(feedback, 5.0, 75.0, prefix="")

            out_refl_water_tif = Path(refl_out_water_tif_base) if (write_water_tiffs and refl_out_water_tif_base) else None
            out_refl_land_tif = Path(refl_out_land_tif_base) if (write_land_tiffs and refl_out_land_tif_base) else None
            out_pixel_water_tif = Path(pixel_out_water_tif_base) if (write_water_tiffs and pixel_out_water_tif_base) else None
            out_pixel_land_tif = Path(pixel_out_land_tif_base) if (write_land_tiffs and pixel_out_land_tif_base) else None
            out_sum_water_tif = Path(sum_out_water_tif_base) if (write_water_tiffs and do_sum and sum_out_water_tif_base) else None
            out_sum_land_tif = Path(sum_out_land_tif_base) if (write_land_tiffs and do_sum and sum_out_land_tif_base) else None

            wR, lR, wP, lP, wS, lS, gt, wkt, written = _process_both_by_acqdate(
                tifs_refl=refl_tifs,
                tifs_pixel=pixel_tifs,
                out_refl_water_tif=out_refl_water_tif,
                out_refl_land_tif=out_refl_land_tif,
                out_pixel_water_tif=out_pixel_water_tif,
                out_pixel_land_tif=out_pixel_land_tif,
                out_sum_water_tif=out_sum_water_tif,
                out_sum_land_tif=out_sum_land_tif,
                bg_rgb=bg_rgb,
                water_ranges=water_ranges,
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
                ("sum_water", sum_water_tif_written),
                ("sum_land", sum_land_tif_written),
            ):
                if written.get(k) is not None:
                    lst.append(Path(written[k]))

            # Validity masks per pipeline
            vR = ((wR + lR) > 0).astype(np.uint8)
            vP = ((wP + lP) > 0).astype(np.uint8)
            vS = None
            if do_sum and (wS is not None) and (lS is not None):
                vS = ((wS + lS) > 0).astype(np.uint8)

            pixel_sizes_m.append(_approx_pixel_size_m_from_gt_wkt(gt, wkt))

            waterR_bin, landR_bin = _water_land_bins_from_counts(wR, lR, vR)
            waterP_bin, landP_bin = _water_land_bins_from_counts(wP, lP, vP)

            if write_water_vec:
                _append_geom(refl_water_geoms, waterR_bin, gt, wkt, "REFL Water")
                _append_geom(pixel_water_geoms, waterP_bin, gt, wkt, "QA_PIXEL Water")

            if write_land_vec:
                _append_geom(refl_land_geoms, landR_bin, gt, wkt, "REFL Land")
                _append_geom(pixel_land_geoms, landP_bin, gt, wkt, "QA_PIXEL Land")

            if do_sum and (wS is not None) and (lS is not None) and (vS is not None):
                waterS_bin, landS_bin = _water_land_bins_from_counts(wS, lS, vS)
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
            out_path: str,
            geoms: list[ogr.Geometry],
            label: str,
            class_id: int,
        ) -> tuple[bool, ogr.Geometry | None]:
            if not geoms:
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
                vec_tasks.append(("REFL Water", refl_out_vec, refl_water_geoms, "Water", 1, self.OUT_REFL_VEC))
            if write_land_vec:
                vec_tasks.append(("REFL Land", refl_out_land_vec, refl_land_geoms, "Land", 2, self.OUT_REFL_LAND_VEC))

        if mode in ("pixel", "both"):
            if write_water_vec:
                vec_tasks.append(("QA_PIXEL Water", pixel_out_vec, pixel_water_geoms, "Water", 1, self.OUT_PIXEL_VEC))
            if write_land_vec:
                vec_tasks.append(("QA_PIXEL Land", pixel_out_land_vec, pixel_land_geoms, "Land", 2, self.OUT_PIXEL_LAND_VEC))

        if mode == "both" and do_sum:
            if write_water_vec:
                vec_tasks.append(("SUM Water", sum_out_vec, sum_water_geoms, "Water", 1, self.OUT_SUM_VEC))
            if write_land_vec:
                vec_tasks.append(("SUM Land", sum_out_land_vec, sum_land_geoms, "Land", 2, self.OUT_SUM_LAND_VEC))

        if vec_tasks:
            # Store finalized geometries (useful for debugging/logging).
            finalized_by_key: dict[str, ogr.Geometry] = {}
            for ti, (desc, out_path, geoms, label, class_id, out_key) in enumerate(vec_tasks, start=1):
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

                wrote, merged = _finalize_and_write_vec(out_path, geoms, label, class_id)

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
        if mode == "both" and do_sum:
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
            if sum_water_tif_written:
                results[self.OUT_SUM_TIF] = str(sum_water_tif_written[0])

            results[self.OUT_REFL_TIF_LIST] = "\n".join(str(pp) for pp in refl_water_tif_written)
            results[self.OUT_PIXEL_TIF_LIST] = "\n".join(str(pp) for pp in pixel_water_tif_written)
            results[self.OUT_SUM_TIF_LIST] = "\n".join(str(pp) for pp in sum_water_tif_written)
        else:
            results[self.OUT_REFL_TIF_LIST] = ""
            results[self.OUT_PIXEL_TIF_LIST] = ""
            results[self.OUT_SUM_TIF_LIST] = ""

        if write_land_tiffs:
            if refl_land_tif_written:
                results[self.OUT_REFL_LAND_TIF] = str(refl_land_tif_written[0])
            if pixel_land_tif_written:
                results[self.OUT_PIXEL_LAND_TIF] = str(pixel_land_tif_written[0])
            if sum_land_tif_written:
                results[self.OUT_SUM_LAND_TIF] = str(sum_land_tif_written[0])

            results[self.OUT_REFL_LAND_TIF_LIST] = "\n".join(str(pp) for pp in refl_land_tif_written)
            results[self.OUT_PIXEL_LAND_TIF_LIST] = "\n".join(str(pp) for pp in pixel_land_tif_written)
            results[self.OUT_SUM_LAND_TIF_LIST] = "\n".join(str(pp) for pp in sum_land_tif_written)
        else:
            results[self.OUT_REFL_LAND_TIF_LIST] = ""
            results[self.OUT_PIXEL_LAND_TIF_LIST] = ""
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
            if mode == "both" and do_sum:
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
